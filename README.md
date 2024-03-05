# ResonanceRopeLibrary
Improve context length generation in large language models by reducing generalization gaps

This is a simple C++ implementation of the RESONANCE ROPE method described in this paper***:

https://huggingface.co/papers/2403.00071 

The key idea is that RESONANCE ROPE reduces the generalization gap for long contexts 
by eliminating interpolation errors in the position embeddings on out-of-distribution positions. 
Combining it with YaRN-style scaling further improves performance.

Here's how you can use these functions:

Get the original RoPE frequencies using get_rope_frequencies(d, base), where d is the model dimension 
and base is the rotary base (e.g. 10000).  Apply RESONANCE ROPE to the original frequencies 
using apply_resonance_rope(orig_thetas, max_len), where max_len is the maximum context length you want to extend to.

Create the RESONANCE ROPE embeddings for the given max_len using get_rope_embeddings(max_len, d, base).
Use these embeddings when initializing the position embeddings of your model.
If you want to combine RESONANCE ROPE with other RoPE scaling techniques like YaRN, 
you'll need to modify the apply_resonance_rope function to incorporate the YaRN scaling strategy.

Note that this is a simplified implementation, and you may need to adjust it based on your specific use case 
and deep learning framework. Additionally, you'll likely need to handle other aspects like finetuning 
and integrating the modified position embeddings into your model architecture.

= = = = = = = = =

*** Here is a simple 'how to' for extending the context window of your programs based on the paper:

Identify the type of position embedding your model uses. The paper focuses on models using Rotary Position Embedding (RoPE).

If using RoPE, apply the RESONANCE ROPE technique proposed in the paper: 
a) Calculate the wavelengths λj of the RoPE features using λj = 2π / θj 
b) Round each wavelength to the nearest integer: λ̃j = round(λj) 
c) Recalculate the new angular frequencies θ̃j = 2π / λ̃j 
d) Replace the original RoPE frequencies Θ = {θ0, θ1, ...} with the new Θ̃ = {θ̃0, θ̃1, ...} 
e) Use the new Θ̃ when computing the RoPE features in your model

Optionally, combine RESONANCE ROPE with another RoPE scaling technique like YaRN: 
a) Apply the YaRN scaling strategy to scale the RoPE wavelengths λ̃j further; and 
b) Use the scaled wavelengths along with RESONANCE ROPE angular frequencies Θ̃

Finetune your model using the updated position embeddings on longer sequence lengths.
The key idea is that RESONANCE ROPE reduces the generalization gap for long contexts 
by eliminating interpolation errors in the position embeddings on out-of-distribution positions. 

Combining it with YaRN-style scaling further improves performance.
