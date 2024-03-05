#include <vector>
#include <cmath>
#include <torch/torch.h>

// Function to calculate original RoPE frequencies
std::vector<double> get_rope_frequencies(int d, double base) {
    std::vector<double> thetas(d / 2);
    for (int j = 0; j < d / 2; j++) {
        thetas[j] = base * std::pow(-2, 2 * j / d);
    }
    return thetas;
}

// Function to apply RESONANCE ROPE
std::vector<double> apply_resonance_rope(const std::vector<double>& orig_thetas, int max_len) {
    std::vector<double> new_thetas(orig_thetas.size());
    for (int i = 0; i < orig_thetas.size(); i++) {
        double wavelength = 2 * M_PI / orig_thetas[i];
        double new_wavelength = std::round(wavelength);
        new_thetas[i] = 2 * M_PI / new_wavelength;
    }
    return new_thetas;
}

// Function to create RoPE embeddings
torch::Tensor get_rope_embeddings(int max_len, int d, double base) {
    auto orig_thetas = get_rope_frequencies(d, base);
    auto new_thetas = apply_resonance_rope(orig_thetas, max_len);

    auto embeddings = torch::zeros({max_len, d});
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d / 2; i++) {
            embeddings[pos][2 * i] = std::sin(pos * new_thetas[i]);
            embeddings[pos][2 * i + 1] = std::cos(pos * new_thetas[i]);
        }
    }
    return embeddings;
}
