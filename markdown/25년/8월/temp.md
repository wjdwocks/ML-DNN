In the single teacher setting, the teacher model is designed to process the Image Representation (IR) derived from the raw time series data. This teacher network is based on a 2D convolutional neural network (CNN) architecture tailored for image inputs. Importantly, the teacher is pretrained and its parameters remain fixed during the knowledge distillation process, serving as a stable source of supervisory signals.

As depicted in Figure X, the teacher model receives the 
64
×
64
64×64 pixel IR and extracts high-level feature representations through multiple convolutional stages followed by fully connected layers. The final output of the teacher network is a set of logits 
𝑧
𝑇
z 
T
​
  representing class predictions.

Conversely, the student model operates directly on the raw one-dimensional time series data, employing a 1D CNN architecture adapted for sequential inputs. This architectural distinction reflects the difference in input modalities and ensures that each model is optimized for its respective data format. The student network outputs logits 
𝑧
𝑆
z 
S
​
  after processing.

During training, the student is optimized by minimizing a composite loss function that combines the standard cross-entropy loss with a knowledge distillation (KD) loss. The KD loss aligns the student’s output distribution with the teacher’s softened logits, enabling effective knowledge transfer despite the differing input modalities.

This framework allows the student model to benefit from the rich, modality-transformed information captured by the pretrained teacher, while maintaining the flexibility of processing raw signals during inference.