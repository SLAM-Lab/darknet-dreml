#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif
#include <assert.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

#ifdef IMG_SEG
    n = 1;
#endif
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
#ifdef IMG_SEG
    l.c = classes;
#else
    l.c = n*(classes + 4 + 1);
#endif
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));

#ifdef IMG_SEG
    l.outputs = l.h*l.w*classes; //one is the actor that if the point is optic or not
#else
    l.outputs = h*w*n*(classes + 4 + 1);
#endif

    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.loss = calloc(l.inputs*batch, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.loss_gpu = cuda_make_array(l.loss, l.inputs*batch);
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

#ifdef IMG_SEG
    fprintf(stderr, "yolo  l.outputs= %d l.c is %d\n", l.outputs, l.c);
#else
    fprintf(stderr, "yolo\n");
#endif
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

#ifdef IMG_SEG	
    l->outputs = h*w;
#else
    l->outputs = h*w*l->n*(l->classes + 4 + 1);
#endif
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

float get_yolo_mask(float *x, int index, int i, int j)
{
    float mask_pred_ij = x[index];
    //b.y = (j + x[index + 1*stride]) / lh;
    //b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    //b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return mask_pred_ij;
}

void delta_yolo_seg(float truth, float pred, int index, int i, int j, int w, int h, float *delta, int stride)
{
      delta[index] = abs(truth - pred);
}

void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat, float *loss)
{
#ifdef IMG_SEG
    int n;
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] =-((n == class)?1 : 0) + output[index + stride*n];
        loss[index + stride*n] = (((n == class)?1 : 0)-class)*log(1-output[index + stride*n]+.0000001);
    }
#else
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
#endif
}

static int entry_index(layer l, int batch, int location, int entry)
{
#ifdef IMG_SEG
    return batch*l.outputs+location*l.classes+entry;
#else
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
#endif
}


#ifdef IMG_SEG
void forward_yolo_layer(const layer l, network net)
{
    int i,j,b;

    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    int k;
    for(k = 0; k < l.w*l.h*2; ++k){
        int index = k;
        l.delta[index] = 0 - l.output[index]; 

    }

    #if 1
    for (b = 0; b < l.batch; ++b){
            //int index = entry_index(l, b, l.w*l.h, 0);
            int index = b*l.w*l.h;
            activate_array(l.output + index, l.w*l.h*2, LOGISTIC);
//            index = entry_index(l, b, l.w*l.h, 1);
//            activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
    }
    #endif

    if(!net.train) return;
    //int count = 0;
    *(l.cost) = 0;
    float largest = -FLT_MAX;

    for(i = 0; i < l.w*l.h; ++i){
        if(net.input[i] > largest) net.input[i] = largest;
    }
    int count_pixel = 0;
    for (b = 0; b < l.batch; b++) {
        for (j = 0; j < l.h*l.w; j++) {
              int class_truth_index = b*l.w*l.h*3+j;
              float class_truth = net.truth[class_truth_index]; //0,1,2,.....19
              int class_index = b*l.w*l.h+j;
	      delta_yolo_class(l.output, l.delta, class_index, class_truth, l.classes, l.w*l.h, 0, l.loss);
            }
        }
    *(l.cost) = pow(mag_array(l.loss, l.outputs * l.batch), 2);
    printf("The number of positive pixel is %d\n", count_pixel);
    printf("YOLO ----- l.cost  %f,  Avg Cost: %f\n", *(l.cost), *(l.cost)/(512*1024));
}
#else

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {

		#ifdef	CUSTOM_BACKPROP
		int activeN = 0;
		#endif

                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }

			#ifdef	CUSTOM_BACKPROP
			int c;
			int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords + 1);

			if(l.output[obj_index] > 0.5)
			{
				l.delta[obj_index] = l.noobject_scale * l.output[obj_index];

				//printf("@@ %i %i %i %f\n",i,j,n,l.output[obj_index]);
				activeN++;

				for(c = 0; c < l.coords; c++)
				{
					l.delta[box_index+(c*l.w*l.h)] *= -1;
				}

				for(c = 0; c < l.classes; ++c)
				{
					int index = class_index+(c*l.w*l.h);

					float prob = l.output[obj_index] * l.output[index];

					if(prob > 0.5) //FIXME: Hardcoded by Kamyar for YOLOv2
					{				
						//printf("@@@ %i %f\n",c,l.output[obj_index] * l.output[index]);
						l.delta[index] = l.class_scale * l.output[index];
					}
					else
					{
						l.delta[index] = 0; //l.class_scale * (0 - l.output[index]);
					}
				}
			}
			else
			{
				l.delta[obj_index] = 0;

				for(c = 0; c < l.coords; c++)
				{
					l.delta[box_index+(c*l.w*l.h)] = 0;
				}

				for(c = 0; c < l.classes; ++c)
				{
					l.delta[class_index+(c*l.w*l.h)] = 0;
				}
			}
			#endif
                }

		#ifdef	CUSTOM_BACKPROP
		/*
		if(activeN>0)
		continue;

		for (n = 0; n < l.n; ++n) 
		{
			int c;
			int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
			int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
			int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords + 1);

			l.delta[obj_index] = 0;

			for(c = 0; c < l.coords; c++)
			{
				l.delta[box_index+(c*l.w*l.h)] = 0;
			}

			for(c = 0; c < l.classes; ++c)
			{
				l.delta[class_index+(c*l.w*l.h)] = 0;
			}
		}
		*/
		#endif

            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat, 0);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}
#endif

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
#ifdef IMG_SEG
    int i,j,num;
    float *predictions = l.output;
//    if (l.batch == 2) avg_flipped_yolo(l);    dont know how it works, but you can try to open it first
    int count = 0;
    for (j = 0; j < l.h; j++) {
        for (i = 0; i < l.w; i++) {
        //int obj_index  = entry_index(l, 0, i, 4);
        //float objectness = predictions[obj_index];
        //if(objectness <= thresh) continue;
//        dets[count].seg = get_yolo_mask(predictions, class_index, l.w, l.h);
        float max_val = 0.5;
        int class_index = entry_index(l, 0, j*l.w+i, 0);
        for(num=0; num<2; num++){

        if(get_yolo_mask(predictions, class_index, l.w, l.h) > max_val){
          dets[count].classes = num;
          max_val = get_yolo_mask(predictions, class_index, l.w, l.h);
        }
        else{
          dets[count].classes = 0;
        }
    }
        if(get_yolo_mask(predictions, class_index, l.w, l.h) > max_val){
        printf("count is %d,class is %d\n",count, dets[count].classes);
      }
      count++;
      }
    }
    //correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;    //count = l.w * l.h
#else
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
#endif
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
#ifdef IMG_SEG
   for (b = 0; b < l.batch; ++b){
	int index = b*l.w*l.h;
	activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
   }
#else
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
#ifdef IMG_SEG
    cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
#endif
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

