## Single Teacher Setting Architecture 예시.
In the single-teacher setting, the teacher network is composed of 2D convolutional layers to process Image Representations (IR) derived from the raw time series, whereas the student network is composed of 1D convolutional layers to process the raw signal data. The IR is either a GAF image or a PI. To ensure consistency across IR types and enable a unified teacher architecture, all IRs are resized to $64 \times 64$ pixels.

As shown in Figure~X(a), the student receives the raw 1D time-series input, while the teacher takes the corresponding 2D Image Representation (e.g., GAF or PI). The teacher is a pretrained 2D-CNN kept frozen during training, and thus serves only to provide soft targets. Learning proceeds with two terms: (i) a cross-entropy (CE) loss between the student's logits $z_S$ and the hard labels, and (ii) a knowledge-distillation (KD) loss that matches the student's softened logits to the teacher's softened logits. The single-teacher training objective is
\[
\mathcal{L}_{\text{single}} = (1-\lambda)\cdot \mathcal{L}_{\mathrm{CE}} + \lambda \cdot \mathcal{L}_{\mathrm{KD}},
\]
where $0 \le \lambda \le 1$ balances the cross-entropy term and the KD term.
As a result, the student—although trained only on raw time-series—learns complementary information distilled from the teacher’s image representations. It therefore benefits from image-domain cues (e.g., global correlations in GAF/PI) without constructing IRs or using extra models at inference.


\subsection{Single-Teacher Setting}
In the single-teacher setting, the student network is composed of 1D CNN to process the raw time-series input. We consider two teacher variants. The IR teacher ($T_{\mathrm{IR}}$) uses 2D CNN to process Image Representations, whereas the signal teacher ($T_{\mathrm{sig}}$) uses 1D CNN to process the raw signal. The IR is either a GAF image or a PI, and all IRs are resized to $64 \times 64$ pixels to ensure consistency and enable a unified 2D teacher architecture.

As shown in Figure~3(a) for the IR-teacher variant, the student receives the raw 1D signal while the teacher takes the corresponding $64 \times 64$ IR (e.g., GAF or PI). In the signal-teacher variant, both the student and the teacher operate on the raw signal, but the teacher serves as a stronger, pretrained model. In either case, the teacher is pretrained and kept frozen during training, providing only soft targets.

Learning proceeds with two terms: (i) a cross-entropy (CE) loss between the student's logits $z_S$ and the hard labels, and (ii) a knowledge-distillation (KD) loss that matches the student's softened logits to the teacher's softened logits. The single-teacher training loss is
\begin{equation}
\mathcal{L}_{\text{single}} = (1-\lambda)\,\mathcal{L}_{\mathrm{CE}} + \lambda\,\mathcal{L}_{\mathrm{KD}},
\end{equation}
where $0 \le \lambda \le 1$ balances the CE and KD terms. Here, $z_S$ and $z_T$ denote the student and teacher logits, respectively.

This setup allows a unimodal student (raw signals only at inference) to benefit from complementary supervision: with $T_{\mathrm{IR}}$, the student absorbs image-domain cues (e.g., global correlations captured by GAF/PI) without constructing IRs at test time; with $T_{\mathrm{sig}}$, the student distills stronger temporal modeling from a higher-capacity signal-side teacher.


\subsection{Single-Teacher Setting}
In the single-teacher setting (Figure~3), the student uses a 1D CNN to process the raw time-series input. We consider two teacher variants: a signal teacher $T_{\mathrm{sig}}$ that operates on the raw signal with a 1D CNN (Figure~3(a)), and an IR teacher $T_{\mathrm{IR}}$ that operates on a Image Representation with a 2D CNN (Figure~3(b)). The IR is either a GAF image or a PI, and all IRs are resized to $64 \times 64$ pixels to enable a unified 2D teacher architecture.

As shown in Figure~3(a), the signal-teacher variant feeds the raw 1D signal to both the student and $T_{\mathrm{sig}}$, with the teacher acting as a stronger, pretrained model. In Figure~3(b), the IR-teacher variant feeds the raw 1D signal to the student and the corresponding $64 \times 64$ IR (e.g., GAF or PI) to $T_{\mathrm{IR}}$. In both cases, the teacher is pretrained and kept frozen during training, providing only soft targets.

The student is learned by minimizing two terms: (i) cross-entropy on hard labels and (ii) a distillation loss from teacher on softened logits. The resulting training loss is
\begin{equation}
\mathcal{L}_{\text{single}} = (1-\lambda)\,\mathcal{L}_{\mathrm{CE}} + \lambda\,\mathcal{L}_{\mathrm{KD}},
\end{equation}
where $0 \le \lambda \le 1$ balances the CE and KD terms.

This setup allows a unimodal student (raw signals only at inference) to benefit from complementary supervision: with $T_{\mathrm{IR}}$, the student absorbs image-domain cues (e.g., global correlations captured by GAF/PI) without constructing IRs at test time; with $T_{\mathrm{sig}}$, the student distills richer temporal representations from a higher-capacity signal-side teacher.



## Persistence Image Extraction by Topological Data Analysis
Topological Data Analysis (TDA) is a mathematical approach designed to extract global topological patterns from complex and high-dimensional data structures[reference]. Specifically, TDA is an approach that studies the shape of data by leveraging concepts from algebraic topology, aiming to quantify and summarize structural information such as connectivity, loops, and voids within datasets. Conventional signal processing methods for time-series data are often limited by their sensitivity to noise and inability to extract global structural features. In contrast, TDA effectively addresses these issues by emphasizing topological characteristics, thereby enabling robust and reliable extraction of global patterns even from noisy data[reference]. Particularly in time-series analysis, TDA effectively captures temporal dynamics and patterns that traditional approaches may overlook, making it suitable for extracting features that are invariant to noise and small perturbations.
%
The application of TDA begins with converting raw one-dimensional signal data into a high-dimensional point cloud representation[reference]. This transformation is typically performed using a sliding window embedding technique, where sequential segments of the signal are extracted and mapped into a higher-dimensional space. Each segment is thus represented as a point reflecting temporal patterns. The collection of these points forms a point cloud that captures the temporal relationships inherent in the original signal.[reference].
%
Next, persistent homology is computed from the point cloud. This step commonly employs the Vietoris-Rips method to build simplicial complexes[reference]. Points within a certain radius are connected, and as this radius gradually increases—a process known as filtration[reference]—topological features such as connected components, loops, and cavities emerge (birth) and disappear (death). The lifespan of these features is summarized in a Persistence Diagram (PD), which effectively encodes the intrinsic structure of the data, thereby providing robustness. However, a limitation of PDs is their irregular and variable size, which hinders direct integration with machine learning algorithms. To address this, PDs are transformed into Persistence Images (PI) by mapping each PD point onto a discretized grid weighted by Gaussian kernels. Concretely, each point in the PD is first transformed into a two-dimensional Gaussian kernel centered at the corresponding birth and persistence coordinates. These kernels are then accumulated on a predefined regular grid, producing a stable, fixed-size vectorized image representation (PI). This results in fixed-size representations that are well-suited for machine learning tasks.


## Multi Teacher Setting
\subsection{Leveraging IR with Multiple Teachers}
In the multi-teacher setting, both the student and Teacher~1 ($T_1$) take the raw 1D time series as input, whereas Teacher~2 ($T_2$) receives the corresponding $64 \times 64$ Image Representation (IR; e.g., GAF images or PImages). Accordingly, the student and $T_1$ are composed from 1D CNN, while $T_2$ uses 2D CNN suited for image inputs. During distillation, both teachers are pretrained and kept frozen only the student is updated.

The student is learned by minimizing two terms: (i) cross-entropy on hard labels and (ii) a distillation loss from each teacher on softened logits. The resulting training loss is
\begin{equation}
\mathcal{L}_{\text{multi}}
= (1-\lambda)\,\mathcal{L}_{\mathrm{CE}}
+ \lambda\Big[(1-\alpha)\,\mathcal{L}_{\mathrm{KD}}(z_{T_1}, z_S)
+ \alpha\,\mathcal{L}_{\mathrm{KD}}(z_{T_2}, z_S)\Big],
\end{equation}
where $0 \le \alpha \le 1$ controls the relative contribution of the raw-signal teacher ($T_1$) and the IR teacher ($T_2$). $z_S$ and $z_{T_k}$ denote the student and the $k$-th teacher logits, respectively.

Because the teachers and the student operate on different modalities and employ different architectures, their internal representations can be misaligned, creating a knowledge gap that weakens distillation. To mitigate this mismatch, we adopt an annealing initialization. Before distillation, we train a proxy model that shares the student's architecture and capacity from scratch on raw signal data using only the cross-entropy loss. The learned weights are then used to initialize the student (instead of random initialization). This warm initialization narrows the representational gap, stabilizes early training, and improves the effectiveness of knowledge transfer from the teachers.

This design lets a unimodal student absorb complementary cues from both sources—richer structural patterns distilled from $T_1$ and $T_2$—while requiring only raw signals at inference time.



\subsection{Leveraging IR with Multiple Teachers}
As shown in Figure~3(c), the student and Teacher~1 ($T_1$) take the raw 1D time series as input, whereas Teacher~2 ($T_2$) receives the corresponding $64 \times 64$ Image Representation (IR; e.g., GAF images or Persistence Images). Accordingly, the student and $T_1$ are composed of 1D convolutional layers, while $T_2$ uses 2D convolutional layers suitable for image inputs. During distillation, both teachers are pretrained and kept frozen only the student is updated.
%
The student is learned by minimizing two terms: (i) cross-entropy on hard labels and (ii) a distillation loss from each teacher on softened logits. The resulting training loss is
\begin{equation}
\mathcal{L}_{\text{multi}}
= (1-\lambda)\,\mathcal{L}_{\mathrm{CE}}
+ \lambda\Big[(1-\alpha)\,\mathcal{L}_{\mathrm{KD}}(z_{T_1}, z_S)
+ \alpha\,\mathcal{L}_{\mathrm{KD}}(z_{T_2}, z_S)\Big],
\end{equation}
where $0 \le \lambda \le 1$ balances $\mathcal{L}_{\mathrm{CE}}$ and distillation term, and $0 \le \alpha \le 1$ controls the relative contributions of the raw-signal teacher ($T_1$) and the IR teacher ($T_2$). Here, $z_S$ and $z_{T_k}$ denote the student and the $k$-th teacher logits.
%
Because the teachers and the student operate on different modalities and employ different architectures, their internal representations can be misaligned, creating a knowledge gap that weakens distillation. To mitigate this mismatch, we adopt an annealing-style initialization. Before distillation, we train a proxy model with the same architecture and capacity as the student from scratch on raw signal data using only the cross-entropy loss; the resulting weights initialize the student instead of random parameters. This warm start narrows the representational gap, stabilizes early training, and improves the effectiveness of knowledge transfer from the teachers.
%
This design lets a unimodal student absorb complementary cues from both sources—capturing richer structural patterns from the raw-signal and image domains—while requiring only raw signals at inference time.
