## Introduction
1. Human activity recognition based on wearable sensor data has been widely applied in domains such as healthcare, medical monitoring, smart homes, and abnormal detection for security. To meet these demands, numerous studies have explored deep learning approaches to enhance the accuracy and robustness of wearable sensor–based activity recognition. However, there are still challenges: raw wearable sensor data have inherently limited information, show variability between individuals in activity patterns, involve difficulty in handling sensor noise, and exhibit high sensitivity to segmentation and sampling strategies.

2. To address these limitations, recent studies have explored transforming raw time-series data from wearable sensors into image-based representations, which have achieved promising results. In particular, persistence images (PI) and gramian angular fields (GAF) provide rich contextual information and capture high-dimensional structures that cannot be obtained from raw data alone. PI encodes topological features that remain invariant under transformations such as stretching, bending, or rotation, thereby reflecting global and intrinsic properties of the data. These features can be extracted through topological data analysis (TDA), which preserves the essential shape characteristics of the time-series. In contrast, GAF represents pairwise similarities between time points by mapping them into polar coordinates, enabling the visualization of temporal correlations and periodic patterns. These image representations effectively capture more complex and informative patterns within time-series. Moreover, they can serve as useful inputs for feature extraction when directly applied to deep neural networks. However, generating IRs requires additional computational time and memory, which can lead to degraded performance on resource-constrained wearable devices.

3. To further address these limitations, knowledge distillation (KD) has been adopted as an effective strategy for building lightweight models. In this paradigm, a smaller student model learns from the guidance of a larger teacher model, thereby distilling its knowledge in a compressed form. Beyond the conventional single-teacher setting, multiple teachers can be employed to provide complementary supervision, producing a more robust student. Moreover, KD can be extended to multimodal teachers, where diverse image-based models transfer knowledge to a time-series student model. Such approaches not only enhance the generalization capability of the student but also improve its robustness against noise. In addition, the supplementary features derived from IRs convey information that raw time-series data cannot capture, thereby enhancing the interpretability of the student through the distillation process. Previous works have examined KD using IRs alongside raw signals, but little attention has been paid to the potential of combining multiple, heterogeneous IRs within a unified distillation framework.

4. In this paper, we investigate knowledge distillation with image representations to develop more effective lightweight models for human activity recognition using wearable sensor data. First, we generate PI and GAF representations and train them from scratch to examine the role and individual performance of each modality. Second, we evaluate various KD strategies that combine IRs with raw time-series data, assessing the effectiveness of each IR-driven distillation setting. Finally, we explore a three-modality setting where PI, GAF, and time-series data are jointly incorporated into the KD process. In addition, we examine both cases with and without feature-based distillation, providing insights into the compatibility of each IR with different KD strategies, as well as the interplay when multiple IRs are used together. Through these experiments, we aim to answer whether greater informational richness consistently leads to better outcomes, offering practical guidance for designing efficient and high-performing systems for wearable sensor analysis. We further demonstrate the performance of distilled lightweight models in terms of both inference time and accuracy, and compare models trained with time-series only, single IR + time-series, and multiple IRs + time-series configurations.

## Background
### TDA
Topological Data Analysis (TDA) is a mathematical framework designed to extract complex and informative topological features from data \cite{}. These complementary features have been exploited in many domains to enhance model performance \cite{}.

To obtain PI from time-series, the process begins with transforming one-dimensional data into a high-dimensional point cloud through a sliding window approach \cite{}. Then, persistence homology is computed on this point cloud. Within this process, points located within a distance $\epsilon$ are connected to construct a simplicial complex. As $\epsilon$ increases, the complex expands and topological structures—such as connected components, loops, and voids—appear and disappear, with their lifespans defined by corresponding birth and death times. The resulting set of birth–death pairs is represented in a persistence diagram (PD), which remains stable under small perturbations of the data \cite{}. However, because PDs vary in size, they cannot be directly applied to deep learning models.

To address this issue, PDs need to be converted into fixed-size vector representations. One such approach is the persistence image (PI). In this method, a persistence surface is first generated by assigning a weighted Gaussian to each point in the PD and combining them. This surface is then discretized over a regular grid, where the value of each grid cell is calculated to produce a pixel-based matrix representation, referred to as the PI.

Through this procedure, PIs capture structural patterns not easily extracted from raw time-series data and provide complementary, informative representations. Nevertheless, generating PIs incurs considerable computational and memory overhead, which makes their direct use impractical on resource-constrained devices such as wearable sensors. To overcome this issue, prior studies have employed knowledge distillation (KD), where a teacher network trained with PIs transfers knowledge to a student model operating solely on time-series data \cite{}. In this way, the student can benefit from the richer information provided in PIs without the need to explicitly compute them. An illustration of a sample time-series signal and its corresponding PD and PI is shown in Figure \ref{figure:PD_PI}.

### GAF
The Gramian Angular Field (GAF) is one of the methods used to transform time-series data into image representations \cite{}. GAF maps each time step of a time series to an angular value in polar coordinates and then constructs a Gramian matrix by computing the cosine of angle sums for every pair of time steps. The resulting GAF image encodes the correlations between different time steps in the sequence. Since the diagonal of the Gramian matrix corresponds to the sum of each time step with itself, it preserves the original information of the raw time series. Moreover, entries near the diagonal capture correlations between adjacent time steps, while those farther away reflect correlations between more distant points. In this way, GAF images provide inter-temporal relationships and additional information that cannot be directly observed from the original time-series data, and are therefore frequently adopted to enhance performance in tasks such as human activity recognition \cite{}.

To construct a GAF image, the time-series values are first normalized into the range $[-1, 1]$. Next, each normalized value is mapped to a polar coordinate using the arccosine function, defined as $\phi_t = \arccos(x_t)$, where $t$ denotes the time step \cite{}. According to Wang et al. \cite{}, the resulting angular values can be encoded in two different ways: by using the cosine of angle sums to form the Gramian Angular Summation Field (GASF), or by using the sine of angle differences to form the Gramian Angular Difference Field (GADF). In GASF, the diagonal entries retain magnitude information from the raw series, thereby preserving the original signal, whereas in GADF, the diagonal entries are zero by definition, discarding this information. In this study, we adopt the GASF formulation.
For two different time steps $i$ and $j$, the GAF entry is defined as:
\begin{equation}
    \text{GAF} =
    \begin{pmatrix}
    \cos(\phi_1 + \phi_1) & \cos(\phi_1 + \phi_2) & \cdots & \cos(\phi_1 + \phi_n) \\
    \cos(\phi_2 + \phi_1) & \cos(\phi_2 + \phi_2) & \cdots & \cos(\phi_2 + \phi_n) \\
    \vdots & \vdots & \ddots & \vdots \\
    \cos(\phi_n + \phi_1) & \cos(\phi_n + \phi_2) & \cdots & \cos(\phi_n + \phi_n)
    \end{pmatrix},
    \label{eq:GAF_matrix}
\end{equation}
where $n$ denotes the resolution of the GAF image. This can be adjusted either by interpolating or by downsampling the original time-series window size.
Moreover, the term $\cos (\phi_i + \phi_j)$ can also be expressed as a Gramian inner product, formulated as:
\begin{equation}
<x_i,x_j> = x_i \cdot x_j - \sqrt{1-x_i ^2} \cdot \sqrt{1- x_j ^2}.
\end{equation}

Figure \ref{figure:GAF_example} visualizes an example of a time-series sequence and its corresponding GAF image. Because we adopt the GASF formulation, the resulting matrix is symmetric with respect to the main diagonal ($y=x$).

### KD
Knowledge Distillation (KD) is one of the widely used techniques for model compression. In KD, a high-capacity teacher network transfers its knowledge to a smaller student network, allowing the student to maintain efficiency while still achieving competitive performance. KD was first introduced by Bucilua \emph{et al.} \cite{} and was later popularized by Hinton \emph{et al.} \cite{}, who demonstrated that using softened teacher outputs with temperature scaling provides richer information than hard labels alone, thereby making distillation broadly effective and widely adopted. In conventional KD, the total training loss is formulated as a combination of the cross-entropy loss between the student’s predictions and the ground-truth labels, and the Kullback–Leibler (KL) divergence loss between the softened output distributions of the teacher and student. The loss is defined as follows:
\begin{equation}
\mathcal{L} = (1 - \lambda)\mathcal{L}_{CE} + \lambda \mathcal{L}_{KD},
\label{equ:Single_KD}
\end{equation}
where, 0 < $\lambda$ < 1 is a hyperparameter that balances the contribution of $\mathcal{L}{CE}$ and $\mathcal{L}_{KD}$. The cross-entropy term is defined as: 
\begin{equation}
\mathcal{L}_{CE} = Q(\sigma(l_S), y)
\end{equation}
where $Q(\cdot)$ denotes the cross-entropy function, $l_S$ is the student’s logits, $y$ is the ground-truth label, and $\sigma(\cdot)$ denotes the softmax operation. The distillation loss is given by:
\begin{equation} \label{kd_loss}
\mathcal{L}_{KD} = \tau^2 \, KL(\sigma(l_T / \tau), \sigma(l_S / \tau)),
\end{equation}
where $l_T$ and $l_S$ are the logits of the teacher and student networks, respectively, and $\tau > 1$ is the temperature parameter used to soften the probability distributions. The KL term measures how closely the student’s output distribution aligns with that of the teacher.

Conventional KD typically relies on logit-based losses that align the output logits of the teacher and the student. However, feature-based losses that align the intermediate representations of both networks have also been explored \cite{}. In feature-based distillation, feature maps are extracted from corresponding intermediate layers of the teacher and student, and then aligned, allowing the student to learn not only the final predictions but also the teacher’s hidden representations. Tung \emph{et al.} \cite{} proposed a feature-based KD method based on pairwise similarity between samples. In their approach, feature maps from intermediate layers are reshaped and converted into similarity maps, which are then used to guide the student’s training. The similarity maps are constructed as follows:
\begin{equation}
G = \hat{A} \cdot \hat{A}^\top, \quad \hat{A} \in \mathbb{R}^{b \times chw},
\end{equation}
where the original feature maps of shape $(b, c, h, w)$ are reshaped into $A \in \mathbb{R}^{b \times chw}$, and $\hat{A}$ denotes the row-normalized version of $A$. The distillation loss is then defined as the mean squared error (MSE) between the teacher’s and student’s similarity maps:
\begin{equation}
\mathcal{L}_{SP}(G_T, G_S) = \frac{1}{b^2} \sum \left| G_T - G_S \right|^2,
\end{equation}
where $G_T$ and $G_S$ are the similarity maps produced by the teacher and student.

KD can also be applied even when the teacher and student differ in model architecture or input modality. In logit-based distillation, the student only needs to match the class probability distribution, which depends solely on the number of classes in the dataset. Similarly, in feature-based distillation using similarity maps, the resulting matrices depend only on the batch size, ensuring that the teacher and student produce similarity maps of the same dimension regardless of their architectures or input modalities.
This property further extends to the multi-teacher setting. When multiple teachers are used, each possibly trained with different architectures or modalities, the student can still learn from all of them. In this case, the overall objective is defined as the weighted sum of the individual KD losses from each teacher:
\begin{equation}
\mathcal{L}_{KD}^{(m)} 
= (1-\lambda)\,\mathcal{L}_{CE}
+ \lambda \sum_{n=1}^{N} \alpha_n \, \mathcal{L}_{KD}^{(T_n)},
\label{equ:KD_multi}
\end{equation}
where $0 < \lambda < 1$ balances the CE loss and the KD loss, $N$ is the number of teachers, $\alpha_n$ is the weight assigned to the $n$-th teacher, and $\mathcal{L}_{KD}^{(T_n)}$ denotes the individual KD loss with teacher $T_n$.
Through this formulation, the student is able to acquire complementary knowledge from multiple heterogeneous teachers, thereby improving both its generalization ability and robustness across diverse input modalities.

## Proposed Method
In this section, we present the proposed multi-modality KD framework, which is designed to effectively transfer knowledge from multiple teachers to a single student model.
Our approach incorporates three complementary modalities: raw time-series, PI, and GAF.
Each modality is handled by a separately trained teacher network, while the student model uses only the time-series as input.
Intermediate feature maps are extracted from all networks and converted into similarity maps, which are compared to compute the feature-based KD loss.
In parallel, the student’s output logits are aligned with those of all teachers to compute the logit-based KD loss. 
The final training objective is defined as a weighted sum of the cross-entropy loss, the logit-based KD loss, and the feature-based KD loss. 
The following subsections detail each component of the framework, including image representation generation, loss formulation and the annealing strategy.

### Extracting Persistence Image
PIs provide topological information that cannot be directly observed from raw data, and have therefore been widely applied in various machine learning tasks and KD frameworks \cite{}. To effectively extract such topological information from time-series data, we employ TDA. Following previous studies \cite{}, we use the Scikit-TDA library and the Ripser package to compute persistent homology and obtain PDs. Each birth–death pair in the PD is then mapped into the birth–persistence space, after which Gaussian kernels are placed on these points to construct a persistence surface. Finally, this surface is discretized to form a Persistence Image. In our experiments, the PI representation is generated with a resolution of $64 \times 64$.

### Extracting Gaussian Angular Field Image
Gramian Angular Fields (GAFs) are frequently employed in time-series based machine learning tasks and KD frameworks because they can visually capture temporal correlations that are difficult to identify from raw time-series alone \cite{}. To generate GAF images, we employed the GramianAngularField module provided by the pyts library. In order to preserve the diagonal information, we extracted the GASF. Specifically, each time step was min–max scaled to the range $[-1, 1]$, and each time step was then mapped to polar coordinates. The GASF was obtained by computing the cosine of the sum of angular values. Finally, each GAF image was generated with a resolution of $64 \times 64$, ensuring compatibility with the network input size used for PI.