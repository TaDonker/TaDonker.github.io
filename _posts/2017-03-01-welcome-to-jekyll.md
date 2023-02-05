---
title: "Positional Encodings for Time Series"
mathjax: true
layout: post
categories: media
---

Transformers for time series forecasting: How do different positional encodings effect the performance

In this article different positional encodings for a time series forecast will be compared. Two absolute positional encodings and a relative positional encoding will be explained and the effect on forecasting accuracy measured. Also, the combination of different encodings with an embedding of time features (temporal embedding) will be analysed.
The experiments are conducted on a traffic dataset, which shows periodic patterns, and the transformer model proposed by  (Li et al., 2020) in “ Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting ” is used. More details can be found in my bachelor thesis (LINK=======)
 
Results upfront:
Using temporal embeddings resulted in a slight performance improvement on a traffic dataset. A relative positional encoding achieved the best results. Also, the results showed, that it is possible to omit traditional positional encodings and only use temporal embeddings.

Transformers for time series forecasting are rising in research interest and achieve state-of-the-art results on many benchmarks. But while the supplementary research of transformers for Natural Language Processing is already built up, for time series forecasting it lacks behind.
The positional encoding is an essential component of any transformer and is heavily researched for language tasks. Nevertheless, for time series forecasting with transformers comparative studies on positional encodings are scarce and the existing research is contradictory to used methods in recent models. 
Therefore, the effects on performance of two commonly used absolute positional encodings and a relative positional encoding will be analysed.

One of the main reasons in the original transformer, beside directly handling distant dependencies, to replace recurrent structures with self-attention is to be able to parallelize computation. Recurrent neural networks are inherently sequential, which impedes parallelization and becomes more critical for longer sequences. In contrast, self-attention operates on calculations of products between matrices and is therefore within a sequence permutation equivariant. This creates a problem as language and maybe even more so, time series, are order dependent. As a solution Vaswani et al. use position encodings. 
A position encoding, first introduced in (Gehring et al., 2017), provides positional information. The position is mapped to a vector of continuous numbers and is added to each element of the input sequence of the encoder and decoder before being processed by the attention mechanism. 

  Attention Matrix		Absolute Position Bias		Relative Position Bias
■(z_1@z_2@z_3 )[■(a_11&a_12&a_13@a_21&a_22&a_23@a_31&a_32&a_33 )]                	■(z_1@z_2@z_3 )[■(p_11&p_12&p_13@p_21&p_22&p_23@p_31&p_32&p_33 )]         	  ■(z_1@z_2@z_3 ) [■(r_0&r_(+1)&r_(+2)@r_(-1)&r_0&r_(+1)@r_(-2)&r_(-1)&r_0 )]
        ■(z_1&〖  z〗_2&  z_3 )  		        ■(z_1&  z_2&  z_3 )  	                        ■(z_1&  z_2&  z_3 )  
Figure 3: Absolute and Relative position biases which in this example encode the position for a sequence of length three. z_i represents the ith element of the sequence as Query or Key and their dot-product defines the attention score a_ij . The Position Biases are added to each element z before the matrix multiplication is calculated and therefore affect the attention score. 
For the Absolute Position Bias, to each element their position in a sequence is added, therefore for example the attention score a_11 of the first element with itself includes the same position encoding twice, once in the Query and once in the Value representation.  
In contrast, the Relative Position Bias encodes the relationships directly and includes the distance to other elements. Therefore, for example r_0 stays the same for each element’s attention score with itself. 
Unlike in time series forecasting research, in NLP positional encodings are explored heavily (Dufter, Schmitt and Schütze, 2021). The research on positional information can be grouped in either absolute or relative encodings. The encodings of the original transformer are absolute and encode each position p from 1 to maximum sequence length into a d-dimensional vector. Hence, a mapping  f∶ N → R^d   is defined. In the original transformer paper by Vaswani et al. two different absolute encodings are investigated, an engineered fixed encoding with sinusoidal waves and one where the encoding is learned completely by the model itself. 

Absolute Positional Encodings

A common solution for an absolute position encoding is a learned embedding. The position of each element within the input sequence is modelled with a learned lookup table and produces the d-dimensional output. An advantage is that the resulting positional encoding is completely data-driven and is possibly able to learn more complex information rather than only about position (Wang and Chen, 2020), which could be especially useful for time series, if temporal information can be incorporated.

An implementation with PyTorch is rather simple by using the built in Embedding class (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
The required hyperparameters are the sequence length, for time series also called window size, and the output dimension of each learned embedding vector, which regulates the expressiveness.

{% highlight ruby %} class TransformerModel(nn.Module):
####
####
    self.position_embed = nn.Embedding(window_len, n_embd)

    def forward(self, x):
        length = x.size(1) # (Batch_size, sequence_length, num of features)
        positions = torch.tensor(torch.arange(length))
        pos_embedding = self.position_embed(positions)

        x = torch.cat((x, pos_embedding), dim=2)

        for block in self.layers: 
#a transformer consists of consecutive attention layers 
            x = block(x)
        return x {% endhighlight %}






The other popular and used in the final version of the original transformer by Vaswani et al. is the sinusoidal encoding (Equation 6). Like in the learned embedding, the sinusoidal function takes the position index p of each element of a sequence as input, but here the function is predefined without learnable parameters. It produces d waves of different frequencies alternating between sine and cosine. 


$$ {PE}_{\left(p,2j\right)}=sin{\ \left(\ \frac{p}{10000^\frac{2j}{d}}\right)} $$


 
$$ {PE}_{(p,2j+1)}=cos{\left(\ \frac{p}{10000^\frac{2j}{d}}\right)}  $$


 (EQ6)a
 

 
Figure 4: The absolute sinusoidal position encoding uses alternating values of sine and cosine. The wavelength is increasing with higher dimensions of the encoding and therefore avoids identical encodings for different positions even for long sequences. 
To implement the sinusoidal encoding the 
http://nlp.seas.harvard.edu/2018/04/03/attention.html

{% highlight ruby %}
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

self.po_embed = embed.PositionalEmbedding(self.emb_num)
{% endhighlight %}

Both absolute encodings were tested in the original transformer and nearly identical results were observed. The sinusoidal version was chosen because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. For time series forecasting this advantage of the sinusoidal encoding is less critical because in contrast to varying sentence length the forecast horizon can be arbitrarily stipulated beforehand.
Therefore, it is not clear which positional encoding should be used and it could even be possible that completely data dependent positional encodings are favourable for time series. Nevertheless, very little research has been done to explore the effects of different positional encodings for time series forecasting.  

Relative Positional Encodings

The absolute encodings are independent of each other and the main goal is to distinguish different positions, while the relationships are not modelled (Wang et al., 2020). 
On the other hand, relative positional encodings (Shaw, Uszkoreit and Vaswani, 2018) encode the relative distance between tokens and represent the pairwise relationships between the current position and other positions.
(Shaw, Uszkoreit and Vaswani, 2018)  propose to incorporate relative positional information parameters on the Key as well as the Value level of the self-attention mechanism.
To implement relative positional encodings, one has to introduce the pairwise relationships a_ij^V  , a_ij^K   ∈ R^d   between input elements x_i and x_(j ). 
a_ij^V  , a_ij^K are edge information which represent the absolute distance between elements modified by learned weight parameters.
To append relative positional information a_ij^V at the Value level Equation 3 is modified to:

$$ z_i=\ \sum_{j=1}^{n}{{a_{ij}(x}_jW_V\ +\ a_{ij}^V)} $$

(EQ7)


And for the Key matrix the pairwise relationships a_ij^K  is added to Equation 5:

$$ e_{ij}=\frac{(x_iW_Q)\ {(x_jW_K\ +\ a_{ij}^K)}^T}{\sqrt d} $$

(EQ8)

Therefore, in contrast to absolute positional encodings, the attention mechanism requires alteration and is repeatedly calculated in every layer. A resulting drawback is the memory complexity of O(L²). But the relative position encoding by (Shaw, Uszkoreit and Vaswani, 2018) was recently subject to more research and several improvements also in regard to efficiency were proposed (Huang et al., 2018, 2020; Chen, 2021; Luo et al., 2021). 
But for this experiment the original relative positional encoding will be used. A nice implementation can be found here by Yining Hong  ( https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py ). She follows Shaws et al. proposition of an efficient implementation by splitting EQ8 into two terms.



Because for time series forecasting the sequence length is constant and fixed, the traditional positional encodings (e.g. absolute and relative) for each sequence of a batch are equal and the clipping mechanism proposed by Shaw et al. to generalize to longer sequences is not needed. 

Temporal Embeddings
Additionally in contrast to NLP with transformers, where a sequence generally starts with the beginning of a new sentence, for time series it may be useful to incorporate temporal information because a subset of the complete time series is randomly sampled and will have different start times (e.g. weekdays).
This is usually done with embeddings. Each temporal feature (e.g. hour of the day, weekday, month, day of month, day of year…) will be encoded in its own lookup table. It is important to only include and encode a feature if it helps the forecast, as otherwise the dimensionality   unnecessarily increases and the model is more likely to overfit.
 
Figure 9: The conventional positional encoding, which defines the order within a sequence, and a temporal embedding are combined. The temporal embedding models the weekday and the hour of the day of each element with learned embeddings and additionally a global timestamp, which defines the point in time on the whole dataset, is added.



 
Figure 12: Median and best results of the 50%-Quantile forecast for the tested positional encodings. The relative positional encoding combined with the temporal embedding (Relative+Temp) performs best, followed by omitting conventional positional encodings and only using the temporal embedding (Temp_only). Combining a learned embedding with the temporal embedding (PosEmb+Temp) performs similar. With a slightly worse median but a larger variance is the sinusoidal encoding enriched with the temporal embedding (Sinus+Temp). The worst results are the learned embedding (PosEmbedding) and the fixed sinusoidal encoding (SinusPE) encodings with temporal information provided as covariates. 




Time Features as Covariates vs Temporal Embedding

The temporal embedding appears to outperform traditional feature preparation. The median results and also the best runs of the absolute encodings without temporal enrichment are worse than all other positional encodings. The increase of computational complexity by learnable embeddings is a drawback but of small effect. Another disadvantage is, that the embeddings need to be designed deliberately and the design has to be tailored to the specific dataset, which comes with additional hyperparameter tuning. But for time series forecasting it is often the case that a model is continuously used and often it will make sense to find the right embedding to improve all future forecasts.
Also, the effect if no time information at all is included was measured, and the decrease was significant (Table1: WithoutTime). Conclusively, time information is crucial for an accurate forecast and the best performing solution is a temporal embedding.

Absolut Encoding: Learnable vs Fixed

The authors intuition was that the learnable positional embedding would be superior to a fixed sinusoidal encoding because it could possibly incorporate more dataset dependent information. But while the median Q50 result was slightly better for a learned embedding together with a temporal embedding (PosEmb+Temp) than the one with sinusoidal waves (Sinus+Temp), the distributions overlap and the best Q50 result of Sinus+Temp was superior to the PosEmb+Temp, and more experiments are needed for a conclusive judgement. 

Relative Position Encoding and Temporal Embeddings Only

The best performing median results are for the relative encoding and the position encoding only depending on temporal embeddings. Which shows that a traditional absolute encoding is not needed and that the constructed temporal embedding can replace these encodings. This is contrary to (Cai et al., 2020) where they tested a very similar temporal encoding but reported a strong performance decrease. Their loss almost doubled, which is unlikely to have only one cause, but in this work better results are noticeable when combining the sinusoidal global timestamp with the weekday and hour embeddings with a linear layer, which in combination with the global encoding seems to produce a better representation. For example, in the presented traffic benchmark the input window is eight days long and therefore, when only hour and day are encoded, a position is not uniquely identifiable. The global timestamp makes each position unique and by combining global timestep, hour and time embedding with learned parameters, the model is able to produce a better representation. 
The relative positional encoding produced the best results and seems promising for future research on time series forecasting. Because the relative embedding parameters are learned autonomously according to the characteristics of the data, it is suitable for time series. Relative positional encodings are based on the distance of the datapoints, but the importance can still be learned. This harmonizes well with time series forecasting with seasonalities, because there, the distance between periodicities is constant and the importance for such distant points is high. Admittedly, the additional improvement to temporal embeddings is only minimal and more experiments are needed.   
In Conclusion, encoding time features into temporal embeddings seems to have the largest effect. Still, relative positional encodings, which are not very well researched for time series forecasting, performed slightly better than all other positional encodings.
But the effect on forecasting accuracy is rather small and other model parameter like the loss function or data preparation had a larger impact on performance.
(For more details read my Bachelor thesis:::::::::::::::::::::::::::) 

