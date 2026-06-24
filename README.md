# CFAT-with-SAH-Optimization-for-Background-Invariant-Dynamic-Facial-Expression-Recognition

3.3 Data Pre-Processing
The frame sequence of each raw clip is broken down into an ordered sequence of frames. Each clip has a different duration across the three benchmarks, so for a fixed number N of frames sampled, the index sampled in the target temporal axis is obtained by a uniform mapping of the original temporal axis, with the final frame repeated in order to maintain the dimensional consistency (Eq. (1)).
F̂i=F⌊i·T/N⌋,i=1,2,…,N                                                             (1)
Unlike other simple index augmentations, they are not equation-numbered because they are label-preserving stochastic augmentations to increase robustness against viewpoint and timing variance inherent in in-the-wild capture. Each resampled frame is resized to 224x224 pixels then channel-normalized using the pre-trained backbone statistics, after which temporal index jitter and horizontal flipping are applied. The emotion categories are one-hot encoded based on the ground-truth emotion categories, as shown in Eq. (2).
yk=1,ifclass=k;0,otherwise                                                         (2)
It is noted that the class imbalance is more pronounced in FERV39k and DFEW, than in MAFW, and the scale of the loss is corrected by dividing it by the total number of samples and the number of classes in the distribution, so that the loss scale does not become unstable when there are very few samples in a minority class.
wk=N/(C·nk)                                                                         (3)
These class weights are not discarded after preprocessing but rather fed straight into the composite objective function to be optimized later, closing a gap in which the weights were not used during optimization.
3.4 Feature Extraction with VideoMAE v2
The goal of this step is to extract rich spatio-temporal representations without using face detection and hand-crafted motion descriptors. The VideoMAE v2 backbone is a transformer-based model that is trained in a self-supervised manner, given that masked video modeling can be applied to learn generalized motion and appearance statistics from a large unlabeled database, thereby achieving feature extractors that feature good generalization to complex backgrounds and limited labeled data. Spatially each preprocessed frame is divided into non-overlapping patches, flattened and then linearly projected into a shared embedding space containing positional encodings to preserve the spatio-temporal ordering, as specified in Eq. (4).
zi=We·Flatten(pi)+be+Epos                                                             (4)
In the pretraining stage, a significant portion of patches is randomly masked, and for these visible patches, the network only processes the encoder; in this process, the network is trained to reconstruct the missing patches and the reconstruction loss is calculated according to the following formula: Eq. (5)
Lrec=(1/|M|)Σi∈M‖p̂i-pi‖2                                                                 (5)
In fine-tuning, all tokens visible and previously masked are then fed into a stack of multi-head self-attention layers, where the computation of attention is done as per the standard scaled dot-product formulation as shown in Eq. (6).
Attention(Q,K,V)=Softmax(QKT/√dk)V                                                (6)
VideoMAE v2 is kept for three reasons: 1) The masked-modeling is useful for transilience across DFEW, FERV39k and MAFW; 2) The transformer encoder can model long-range temporal dependencies which are crucial for modeling evolving expressions; 3) The high masking ratio puts pressure on the network to focus on global motion semantics instead of the local background appearance, which is directly helpful for the background-invariance objective followed by the following modules.
3.5 Motion-Guided Soft Segmentation
Facial expressions are inherently dynamic in nature as opposed to static appearance, and in-the-wild video introduces spurious variation in lighting, scene and emotion labels. To limit this interference, but without an explicit face detection, a Motion-Guided Soft Segmentation (MGSS) module is developed based on the fact that the inter-frame variation of the dynamic regions of the face is significantly higher than that of the static contextual regions. Normalized for numerical stability and cross-sample comparability, the magnitude of motion M between two frames is a sum of the absolute difference of the two frames, as shown in Eq. (7) and accumulated over the clip.
M̂=M/max(M)                                                                        (7)
This normalized motion map is then passed to a learnable convolutional refinement layer, which maps it to a soft thresholding mask between 0 and 1, as given in Eq. (8) that is unlike hard thresholding and maintains the subtle motion cues found in micro-expressions which are low in amplitude.
S=σ(Wm*M̂+bm)                                                                    (8)
To extract expressive dynamics from the backbone features without using contextual scene information, the mask is used multiplicatively to break the backbone features into complementary foreground and background branches, as described in Eq. (9).
Ftfg=Ft⊙S,Ftbg=Ft⊙(1-S)                                                       (9)
This decomposition is robust to head pose and does not involve face detection or alignment, can be directly applied to raw video frames, and it preserves informative body-gesture cues that are unharmed by occlusion and extreme head pose, offering a principled foundation for the causal disentanglement carried out in the next module.
 
Figure 2. Working process of Segmentation
3.6 Causal Disentanglement Module
While the function of MGSS is to distinguish the foreground and background elements, there may still be some correlations between the contextual scene elements and the emotion labels for in-the-wild data, enabling the network to make use of spurious shortcuts instead of genuine expressive cues. The Causal Disentanglement Module (CDM) seeks to tackle this by adding three complementary regularizers. First, an intervention-based consistency loss randomly shuffles background features within a batch and punishes any variation in the prediction of a model that depends on foreground dynamics (as in Eq. (10)), producing a random background intervention.First, it performs a random intervention over the background by randomly shuffling background features within a batch and punishes any difference in the prediction made by a model that depends on such foreground dynamics (as in Eq. (10)), which creates a random intervention over the background.
Lint=‖g(Ffg,Fbg)-g(Ffg,π(Fbg))‖2                                                  (10)
Second, a corrected orthogonality constraint, which is consistently defined throughout the framework in terms of Frobenius norm of the cross-branch inner product in Eq. (11), is used to ensure the geometric independence of the foreground and background subspaces, which was not the case in the previous formulation where an orthogonality term was introduced twice, under the same symbol but under two different mathematical expressions, once in the segmentation stage and again in the disentanglement stage.
Lorth=‖FfgTFbg‖F2                                                                  (11)
Third, there is a variational upper bound on the mutual information between the background features and the emotion label (in Eq. (12)) that prevents the background branch from including any information about the emotion label.
LMI =I(Fbg;Y)≤ 𝔼[ log q(Y|Fbg)]                                                 (12)
All the three regularizers are fused together into causal regularization loss with balancing hyperparameters as described in Eq. (13).
Lcausal=λ1Lint+λ2Lorth+λ3LMI                                                (13)
These constraints act together to shift the network beyond empirical risk minimization to go over an explicit approximation of causal structure, which leads to better generalization over scene distributions that are different from the ones used for training.
3.7 Spatial-Relational Graph Convolutional Module (SR-GCN) 
The disentangled foreground representation helps reduce background interference but does not explicitly encode structural interactions between various sub-regions of the face, for example, between the brow, eye and mouth areas that are found in many compound and minority class expressions. In order to model this without the need for face detection and landmark localization, the feature map generated by MGSS is divided into a fixed number of spatial sub-regions, and the descriptor of each sub-region is computed as a pooling of the corresponding features reinforced in the foreground. An adjacency matrix is created from a combination of spatial proximity and appearance similarity of node pairs and a gating term, derived from a motion-similarity function, which reduces the weights of edges between regions that have dissimilar temporal dynamics (as specified in Eq. (14)).
Aij=exp(-‖vi-vj‖2/2σ2)·mij                                                       (14)
The resulting graph is then passed through stacked graph convolutional layers that encode information across structurally related regions using the symmetric normalized adjacency (Eq. (15)), so that each region descriptor can be refined based on the information of its neighborhood, instead of being treated alone.
Hgcn(l+1)=σ(D-1/2AD-1/2Hgcn(l)Wgcn(l))                                        (15)
The final-layer node representations are then pooled across the entire graph using pooling in the final layer (Eq. (16)), to obtain a graph-level relational embedding. 
zgcn=Pool(Hgcn(L))                                                             (16)
In practice, the foreground map was divided into a medium-sized fixed grid of regions per frame and the resulting graph is small enough for efficient propagation and still captures the main expressive regions of the face like the upper or lower parts of the face. This module does not rely on landmark detection in order to generate the spatial partitions that are significant for the foreground map, but rather on the existing spatial partitions, while at the same time providing explicit, structurally meaningful relational cues that augment the temporal modeling carried out along the other pathways, especially for the overlapping and subtle emotions of fear and disgust.
3.8 Frequency-Adaptive Transformer (FAT)
The classical methods of dynamic facial expression recognition usually use fixed time sampling, which is not suitable for the micro-expression with the very short duration and the macro-expression with the relatively long duration. The Frequency-Adaptive Transformer overcomes this drawback by implementing three interacting mechanisms. First, the receptive field is not fixed, but rather a small projection network is trained on the input to learn the temporal stride directly from the token sequence as in Eq. (17), enabling the receptive field to vary with expression speed and clip length.
rk=fθ(H)                                                                              (17)
Second, multiple attention heads work at different dilation scales, where the learned rates are used in computing the attention through the scaled dot-product formulation in Eq. (18) to capture both short-term and long-term dependencies.
Attentionk(Q,Rk,V)=Softmax(QRkT/√dk)V                                              (18)
Third, a frequency-gating mechanism is applied using sigmoid-activated pooling to compute a learned importance score for each scale and combines the scale-specific representations into the final frequency-adaptive output (Eq. (19)), which dynamically biases the temporal frequencies that are relevant to the input clip, rather than treating all scales equally.
gk=σ(Wg·Pool(Hk)) , HFAT=Σk=1Kgk⊙Hk                                         (19)
Together, these mechanisms enable the Frequency-Adaptive Transformer to capture both micro and macro expressions within a single, reconfigurable attention pathway, adapting to subtle, high-frequency expressions and capturing the overall expression trend.
3.9 Selective State-Space Temporal Encoder (Mamba-TE) 
The computational complexity of the Frequency-Adaptive Transformer is quadratic in the sequence length, similar to that of pipelines for face detection.The framework has highlighted that pipelines with face detection have quadratic complexity, which can be a practical limitation when dealing with longer clips and resource-constrained deployment, as reported by the Frequency-Adaptive Transformer. To complement the FAT model, a Selective State-Space Temporal Encoder (Mamba-TE) is introduced as a second new model working in parallel on the disentangled foreground token sequence to achieve an efficient alternative with linear complexity for long-range temporal reasoning. A low weight selection network generates discretization and projection parameters directly from the current token as in Eq. (20) and the encoder, instead of using a fixed input-independent recurrence, adapts its action based on the local expression dynamics.
[Δt,Bt,Ct]=sθ(xt)                                                                    (20)
These parameters regulate a linear state recurrence for propagating information across the entire sequence with constant per-step cost, as defined in Eq. (21), and the hidden state is projected to the output representation by a learned readout, as defined in Eq. (22).
htssm=Āht-1ssm+B̄txt                                                              (21)
ytssm=Cthtssm+Dxt                                                                 (22)
The selection mechanism is input-dependent, which allows the encoder to selectively preserve informative motion cues, such as the start of a fear or disgust expression, while discarding segments with redundant information or neutral expressions within a sequence, complementing the attention-based Frequency-Adaptive Transformer to provide a sequence-length-efficient encoding.
Algorithm 1 describes the entire forward pass of CFAT++ for one training batch, showing the data flow from raw video sampling to VideoMAE v2 encoding, motion-guided segmentation, causal disentanglement, the SR-GCN and Mamba-TE pathways, tri-path fusion, classification, and loss computation, where the arrows and boxes indicate the data flow and the equations highlighted above provide meaning for each step.
Algorithm 1: End-to-End Forward Pass and Loss Computation of CFAT++
Input: video batch {V_b}, labels {y_b}, parameters theta
Output: total loss L_total
1:  for each clip V_b do
2:      Sample, augment, normalize  -> X_b                 (Eq. 1)
3:      Encode labels; compute class weights w_k            (Eq. 2,3)
4:      Patch-embed + positional encoding                   (Eq. 4)
5:      VideoMAE v2 encoding; get L_rec                     (Eq. 5,6)
6:      Motion map, soft mask S                             (Eq. 7,8)
7:      Decompose: F_fg, F_bg                               (Eq. 9)
8:      L_int, L_orth, L_MI -> L_causal                     (Eq. 10-13)
9:      Region graph; SR-GCN propagation -> z_gcn           (Eq. 14-16)
10:     Learn dilation r_k; multi-scale attention            (Eq. 17,18)
11:     Frequency gating -> H_FAT                           (Eq. 19)
12:     Selection params; SSM recurrence -> y_ssm            (Eq. 20-22)
13:     Gate-fuse H_FAT, z_gcn, y_ssm -> H_fused             (Eq. 23,24)
14:     Confidence-weighted aggregation -> z                (Eq. 25)
15:     Classify: logits o, probabilities y-hat              (Eq. 26,27)
16:     Compute L_total                                      (Eq. 30)
17: end for
18: theta <- SAHGO_Update(theta, grad L_total)              (Algorithm 2)
19: return theta

3.9.1 Tri-Path Gated Fusion
The output of SR-GCN is a relational embedding, the output of FAT is a multi-scale attention, and the output of Mamba-TE is a long-range sequential representation, all representing different aspects of the original expression, structural, attention-based temporal, and recurrence-based temporal, respectively, and therefore must be jointly combined for classification. A lightweight gating network is used to take the concatenation of the three pathway outputs and output three normalized gate weights via a softmax projection (Eq. (23)).
[g1,g2,g3]=Softmax(Wφ[HFAT;zgcn;yssm])                                            (23)
The last fused representation is a gate-weighted sum of the three pathway embeddings defined in Eq. (24).
Hfused=g1⊙HFAT+g2⊙zgcn+g3⊙yssm                                              (24)
The network can flexibly select either the relational pathway or the sequential pathway for a specific clip, depending on the type of expression being used in that clip, for instance, using the relational pathway for compound expressions with high spatial co-activation, and the sequential pathway for clips with subtle, but temporally extended motion, rather than on a fixed, hand-tuned combination rule.
 
Figure 3. Proposed CFAT + SAHGO Framework
Expression Detection and Classification
However, not all temporal locations are equally important when making a decision because some segments can be neutral or transitional. In this way, a temporal confidence score is therefore learned per token of the fused sequence, and the resulting aggregation along the sequence is weighted by the confidences, to form the single representation of a clip, as described in Eq. (25).
ct=σ(WcHfused,t) ,  z=ΣtctHfused,t                                                    (25)
The fully connected layer (see Eq. (26)) is used to map this aggregated representation to the emotion classification space, and the logits are transformed to class probabilities by the softmax function (see Eq. (27)).
o=Wfz+bf                                                                       (26)
ŷk=eok/Σjeoj                                                                     (27)
It is a structured pipeline, using both confidence-weighted aggregation and dual-temporal fusion to combine the relational and dual-temporal features and generating an interpretable and numerically stable prediction of the emotions from the raw aggregated and fused features.
3.10 Optimization Strategy: Sharpness-Aware Hypergradient Optimization (SAHGO)
In the proposed framework, both the loss function and the optimization method for minimizing it are responsible for generalization. The proposed method in this paper, called Sharpness-Aware Hypergradient Optimization (SAHGO), combines three complementary mechanisms. First, the perturbation step was towards the direction of locally maximized loss (and only the perturbation step was explicitly given in the previous formulation in the literature), whereas here the direction of the outer parameter update, i.e., the gradient, is also calculated towards this direction, as in Eq. (28) we present both the perturbation and the resulting outer step in the same equation.
θ'=θ+ρ∇θL/‖∇θL‖, θt+1=θt-ηt∇θL(θ')                                           (28)
The second is a hypergradient mechanism that dynamically adjusts the learning rate based on the local curvature of the loss function, which helps to prevent oscillations when training, and the Lookahead mechanism that slowly updates a copy of the parameters that continues to interpolate between the updated parameters and the previous ones; both of them are introduced together in Eq. (29) because they act on the same parameters trajectory at each optimization step.
ηt+1=ηt-α∂L/∂ηt, θslow=θslow+β(θfast-θslow)                                      (29)
Finally, the total training goal, which was previously split over different sections, is explicitly consolidated into a single composite loss, as shown in Eq. (30), including the class-balanced classification loss, the VideoMAE reconstruction loss, and the causal regularization loss, providing one goal to minimize instead of several losses separately described.
Ltotal=Lcls+λrecLrec+Lcausal,                                                     (30)
The SAHGO parameter update is called at the end of each batch, as detailed in Algorithm 2, where the sharpness-aware perturbation step, the hypergradient learning-rate adaptation step and the Lookahead stabilization step are extended to a sequential process.
Algorithm 2: SAHGO Parameter Update
Input: theta_t, eta_t, theta_slow, grad_theta L_total, hyperparameters rho, alpha, beta
Output: updated theta_(t+1), eta_(t+1), theta_slow
1:  Compute perturbation: theta' <- theta_t + rho * grad_theta L / ||grad_theta L||   (Eq. 28)
2:  Compute perturbed gradient grad_theta L(theta')
3:  Update fast weights: theta_fast <- theta_t - eta_t * grad_theta L(theta')         (Eq. 28)
4:  Update learning rate: eta_(t+1) <- eta_t - alpha * dL/d_eta_t                    (Eq. 29)
5:  Update slow weights: theta_slow <- theta_slow + beta*(theta_fast - theta_slow)   (Eq. 29)
6:  Set theta_(t+1) <- theta_slow
7:  return theta_(t+1), eta_(t+1), theta_slow

The improved methodology is additive, and corrective, in two distinct ways, from the original formulation. Additively, it presents the Spatial-Relational Graph Convolutional Module, which generates explicit structural reasoning over dynamic facial sub-regions without adding any new face detection step, and the Selective State-Space Temporal Encoder, which is a linear-complexity, recurrence-based alternative to attention for modeling long video sequences that are not appended as an afterthought but rather integrated with the existing Frequency-Adaptive Transformer by a learned Tri-Path Gated Fusion mechanism. The class-balanced weighting term, which was previously computed but never used anywhere in the loss expression, is now directly in the composite objective in Eq. (30), and the index of repeated equations, which appeared twice both in the loss and in the segmentation section, has been removed due to the present continuous renumbering from Eq. (1) to Eq. (30). These additions and corrections maintain the original "causal, frequency adaptive, face detection free" design approach while simultaneously providing complete self-consistency and explicit traceability of the training objective and optimization procedure from raw video input to parameter update.
