---
title: Breast Ultrasound Lesion Analysis
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Research prototype (Master's thesis): dual-stage deep learning for breast ultrasound. A segmentation
model outlines the lesion, a classifier scores how suspicious it looks, Grad-CAM / LIME / SHAP show why, and a
language model explains the result in plain language (set the Space secret `ANTHROPIC_API_KEY`; without it a
template text is used).

**Not a medical device. Not for diagnosis or treatment decisions.**
