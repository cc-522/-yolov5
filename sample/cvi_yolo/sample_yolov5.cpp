#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cviruntime.h>

static const char *kClassNames[8] = {
    "car", "van", "truck", "pedestrian",
    "person_sitting", "cyclist", "tram", "misc"
};

static const float kAnchors[3][3][2] = {
    {{10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f}},
    {{30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f}},
    {{116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f}}
};

static const int kStrides[3] = {8, 16, 32};
static const int kNumClasses = 8;
static const int kInputW = 640;
static const int kInputH = 640;

static const char *kBoxNames[3] = {
    "output0_Gather",
    "select_3_Gather",
    "select_6_Gather"
};
static const char *kObjNames[3] = {
    "select_1_Gather",
    "select_4_Gather",
    "select_7_Gather"
};
static const char *kClsNames[3] = {
    "select_2_Gather",
    "select_5_Gather",
    "select_8_Gather"
};

struct LetterboxInfo {
  float scale;
  int new_w;
  int new_h;
  int pad_x;
  int pad_y;
};

struct Detection {
  float x1, y1, x2, y2;
  float score;
  int cls;
};

static inline float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

static float iou_xyxy(const Detection &a, const Detection &b) {
  float xx1 = std::max(a.x1, b.x1);
  float yy1 = std::max(a.y1, b.y1);
  float xx2 = std::min(a.x2, b.x2);
  float yy2 = std::min(a.y2, b.y2);

  float w = std::max(0.0f, xx2 - xx1);
  float h = std::max(0.0f, yy2 - yy1);
  float inter = w * h;

  float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  float uni = area_a + area_b - inter;
  if (uni <= 0.0f) return 0.0f;
  return inter / uni;
}

static void nms_per_class(std::vector<Detection> &dets, float nms_thresh) {
  std::sort(dets.begin(), dets.end(),
            [](const Detection &a, const Detection &b) { return a.score > b.score; });

  std::vector<int> keep(dets.size(), 1);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (!keep[i]) continue;
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (!keep[j]) continue;
      if (dets[i].cls != dets[j].cls) continue;
      if (iou_xyxy(dets[i], dets[j]) > nms_thresh) {
        keep[j] = 0;
      }
    }
  }

  std::vector<Detection> out;
  out.reserve(dets.size());
  for (size_t i = 0; i < dets.size(); ++i) {
    if (keep[i]) out.push_back(dets[i]);
  }
  dets.swap(out);
}

static LetterboxInfo letterbox_rgb(const cv::Mat &bgr, cv::Mat &rgb_out) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  float r = std::min((float)kInputW / (float)rgb.cols, (float)kInputH / (float)rgb.rows);
  int new_w = (int)std::round(rgb.cols * r);
  int new_h = (int)std::round(rgb.rows * r);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(new_w, new_h));

  rgb_out = cv::Mat(kInputH, kInputW, CV_8UC3, cv::Scalar(0, 0, 0));
  int pad_x = (kInputW - new_w) / 2;
  int pad_y = (kInputH - new_h) / 2;
  resized.copyTo(rgb_out(cv::Rect(pad_x, pad_y, new_w, new_h)));

  LetterboxInfo info;
  info.scale = r;
  info.new_w = new_w;
  info.new_h = new_h;
  info.pad_x = pad_x;
  info.pad_y = pad_y;
  return info;
}

static void fill_input_tensor(CVI_TENSOR *input, const cv::Mat &rgb_letterbox) {
  float qscale = CVI_NN_TensorQuantScale(input);
  int8_t *ptr = (int8_t *)CVI_NN_TensorPtr(input);

  cv::Mat ch[3];
  cv::split(rgb_letterbox, ch);

  int channel_size = kInputH * kInputW;
  for (int c = 0; c < 3; ++c) {
    cv::Mat q;
    ch[c].convertTo(q, CV_8SC1, qscale / 255.0);
    memcpy(ptr + c * channel_size, q.data, channel_size);
  }
}

static inline float tensor_at(CVI_TENSOR *t, int a, int h, int w, int c) {
  float qscale = CVI_NN_TensorQuantScale(t);
  const int8_t *ptr = (const int8_t *)CVI_NN_TensorPtr(t);
  CVI_SHAPE s = CVI_NN_TensorShape(t);

  int A = s.dim[0];
  int H = s.dim[1];
  int W = s.dim[2];
  int C = s.dim[3];

  if (a >= A || h >= H || w >= W || c >= C) return 0.0f;

  size_t idx = ((size_t)a * H * W * C) + ((size_t)h * W * C) + ((size_t)w * C) + c;
  return (float)ptr[idx] * qscale;
}

static void decode_one_scale(
    CVI_TENSOR *box_t,
    CVI_TENSOR *obj_t,
    CVI_TENSOR *cls_t,
    int stride,
    const float anchors[3][2],
    const LetterboxInfo &lb,
    int orig_w,
    int orig_h,
    float conf_thresh,
    std::vector<Detection> &out) {
  CVI_SHAPE s = CVI_NN_TensorShape(box_t);

  int A = s.dim[0];
  int H = s.dim[1];
  int W = s.dim[2];

  for (int a = 0; a < A; ++a) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float tx = tensor_at(box_t, a, y, x, 0);
        float ty = tensor_at(box_t, a, y, x, 1);
        float tw = tensor_at(box_t, a, y, x, 2);
        float th = tensor_at(box_t, a, y, x, 3);

        float tobj = tensor_at(obj_t, a, y, x, 0);
        float obj = sigmoidf_fast(tobj);

        int best_cls = -1;
        float best_cls_score = 0.0f;
        for (int c = 0; c < kNumClasses; ++c) {
          float tcls = tensor_at(cls_t, a, y, x, c);
          float cls_prob = sigmoidf_fast(tcls);
          if (cls_prob > best_cls_score) {
            best_cls_score = cls_prob;
            best_cls = c;
          }
        }

        float score = obj * best_cls_score;
        if (score < conf_thresh) continue;

        float bx = (sigmoidf_fast(tx) * 2.0f - 0.5f + (float)x) * stride;
        float by = (sigmoidf_fast(ty) * 2.0f - 0.5f + (float)y) * stride;
        float bw = std::pow(sigmoidf_fast(tw) * 2.0f, 2.0f) * anchors[a][0];
        float bh = std::pow(sigmoidf_fast(th) * 2.0f, 2.0f) * anchors[a][1];

        float x1 = bx - bw * 0.5f;
        float y1 = by - bh * 0.5f;
        float x2 = bx + bw * 0.5f;
        float y2 = by + bh * 0.5f;

        x1 = (x1 - lb.pad_x) / lb.scale;
        y1 = (y1 - lb.pad_y) / lb.scale;
        x2 = (x2 - lb.pad_x) / lb.scale;
        y2 = (y2 - lb.pad_y) / lb.scale;

        x1 = std::max(0.0f, std::min(x1, (float)orig_w));
        y1 = std::max(0.0f, std::min(y1, (float)orig_h));
        x2 = std::max(0.0f, std::min(x2, (float)orig_w));
        y2 = std::max(0.0f, std::min(y2, (float)orig_h));

        float box_w = x2 - x1;
        float box_h = y2 - y1;
        if (box_w <= 2.0f || box_h <= 2.0f) continue;
        if (box_w > orig_w * 0.9f && box_h > orig_h * 0.6f) continue;

        Detection d;
        d.x1 = x1;
        d.y1 = y1;
        d.x2 = x2;
        d.y2 = y2;
        d.score = score;
        d.cls = best_cls;
        out.push_back(d);
      }
    }
  }
}

static void print_tensor_info(const char *tag, CVI_TENSOR *t) {
  CVI_SHAPE s = CVI_NN_TensorShape(t);
  printf("%s: name=%s, shape=(", tag, t->name ? t->name : "null");
  for (size_t i = 0; i < s.dim_size; ++i) {
    printf("%d%s", s.dim[i], (i + 1 == s.dim_size) ? "" : ",");
  }
  printf("), qscale=%f\n", CVI_NN_TensorQuantScale(t));
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <cvimodel> <image> [conf_thresh] [nms_thresh] [save_path]\n", argv[0]);
    return -1;
  }

  const char *model_file = argv[1];
  const char *image_file = argv[2];
  float conf_thresh = (argc > 3) ? (float)atof(argv[3]) : 0.25f;
  float nms_thresh = (argc > 4) ? (float)atof(argv[4]) : 0.45f;
  const char *save_path = (argc > 5) ? argv[5] : "result.jpg";

  CVI_MODEL_HANDLE model = nullptr;
  int ret = CVI_NN_RegisterModel(model_file, &model);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    return -1;
  }
  printf("CVI_NN_RegisterModel succeeded\n");

  CVI_TENSOR *input_tensors = nullptr;
  CVI_TENSOR *output_tensors = nullptr;
  int32_t input_num = 0;
  int32_t output_num = 0;
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
  if (!input) {
    printf("Cannot find default input tensor\n");
    CVI_NN_CleanupModel(model);
    return -1;
  }

  CVI_TENSOR *box_t[3] = {nullptr, nullptr, nullptr};
  CVI_TENSOR *obj_t[3] = {nullptr, nullptr, nullptr};
  CVI_TENSOR *cls_t[3] = {nullptr, nullptr, nullptr};

  for (int i = 0; i < 3; ++i) {
    box_t[i] = CVI_NN_GetTensorByName(kBoxNames[i], output_tensors, output_num);
    obj_t[i] = CVI_NN_GetTensorByName(kObjNames[i], output_tensors, output_num);
    cls_t[i] = CVI_NN_GetTensorByName(kClsNames[i], output_tensors, output_num);
    if (!box_t[i] || !obj_t[i] || !cls_t[i]) {
      printf("Cannot find output tensors for scale %d\n", i);
      for (int k = 0; k < output_num; ++k) {
        printf("available output[%d] name=%s\n", k, output_tensors[k].name);
      }
      CVI_NN_CleanupModel(model);
      return -1;
    }
  }

  print_tensor_info("input", input);
  for (int i = 0; i < 3; ++i) {
    print_tensor_info(kBoxNames[i], box_t[i]);
    print_tensor_info(kObjNames[i], obj_t[i]);
    print_tensor_info(kClsNames[i], cls_t[i]);
  }

  cv::Mat image = cv::imread(image_file);
  if (image.empty()) {
    printf("Could not open image: %s\n", image_file);
    CVI_NN_CleanupModel(model);
    return -1;
  }

  int orig_w = image.cols;
  int orig_h = image.rows;

  cv::Mat rgb_letterbox;
  LetterboxInfo lb = letterbox_rgb(image, rgb_letterbox);
  fill_input_tensor(input, rgb_letterbox);

  ret = CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_Forward failed, err %d\n", ret);
    CVI_NN_CleanupModel(model);
    return -1;
  }
  printf("CVI_NN_Forward succeeded\n");

  std::vector<Detection> dets;
  decode_one_scale(box_t[0], obj_t[0], cls_t[0], kStrides[0], kAnchors[0],
                   lb, orig_w, orig_h, conf_thresh, dets);
  decode_one_scale(box_t[1], obj_t[1], cls_t[1], kStrides[1], kAnchors[1],
                   lb, orig_w, orig_h, conf_thresh, dets);
  decode_one_scale(box_t[2], obj_t[2], cls_t[2], kStrides[2], kAnchors[2],
                   lb, orig_w, orig_h, conf_thresh, dets);

  nms_per_class(dets, nms_thresh);

  printf("------\n");
  printf("%zu objects detected\n", dets.size());
  for (size_t i = 0; i < dets.size(); ++i) {
    const Detection &d = dets[i];
    printf("det %zu: [%f %f %f %f] score=%f cls=%d (%s)\n",
           i, d.x1, d.y1, d.x2, d.y2, d.score, d.cls,
           (d.cls >= 0 && d.cls < kNumClasses) ? kClassNames[d.cls] : "unknown");
  }
  printf("------\n");

  cv::Mat vis = image.clone();
  for (size_t i = 0; i < dets.size(); ++i) {
    const Detection &d = dets[i];
    cv::rectangle(vis,
                  cv::Point((int)d.x1, (int)d.y1),
                  cv::Point((int)d.x2, (int)d.y2),
                  cv::Scalar(0, 255, 0), 2);

    char text[128];
    snprintf(text, sizeof(text), "%s %.2f",
             (d.cls >= 0 && d.cls < kNumClasses) ? kClassNames[d.cls] : "unknown",
             d.score);
    cv::putText(vis, text,
                cv::Point((int)d.x1, std::max(0, (int)d.y1 - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
  }
  if (!cv::imwrite(save_path, vis)) {
    printf("failed to save result image: %s\n", save_path);
  } else {
    printf("saved result image: %s\n", save_path);
  }

  CVI_NN_CleanupModel(model);
  printf("CVI_NN_CleanupModel succeeded\n");
  return 0;
}

