# FederatedRecon: GAN-Based Reconstruction Attack in Federated Learning

This project demonstrates a proof-of-concept **GAN-based reconstruction attack** in a federated learning setup. A **malicious client** uses gradient updates from the global model to reconstruct private data (e.g., handwritten digit “9” or celebrity faces) that it has never directly accessed.

## 🚀 Core Components
- **Federated Learning Simulation**  
  Built with PyTorch and Flower, simulating two or more clients training on disjoint data subsets (e.g., MNIST digits 0–4 vs. 5–9).
- **Adversarial GAN Client**  
  The malicious client trains a conditional GAN targeted on a specific class label using **confidence** and **diversity** losses, with **delayed training** and **feature-level guidance**.
- **Reconstructions on MNIST & CelebA**  
  Shows how realistic reconstructions emerge, including faint facial outlines on CelebA without ever seeing the actual celebrity images.

## 📄 Full Walkthrough
For a detailed explanation, illustrations, and experimental results, read the article on Medium:
**"Federated Recon: GAN‑Based Reconstruction Attack In A Federated Learning Setup"**  
➤ https://medium.com/@lost_particles/federated-recon-gan-based-reconstruction-attack-in-a-federated-learning-setup-4545672162e3

## 💡 Why It Matters
Federated learning is often seen as a privacy-first approach—but this work highlights that **shared gradients can leak sensitive data**. Understanding GAN-based inversion attacks helps underscore the need for rigorous privacy defenses.

## 🚧 Next Steps
- Explore and implement defenses (e.g., differential privacy, secure aggregation).
- Extend experiments with more clients or complex datasets.
- Fine-tune GAN losses and training strategies for stronger reconstructions.

---

## 🧩 Installation & Usage

```bash
git clone https://github.com/lost-particles/federatedRecon.git
cd federatedRecon

# Install dependencies (Flower, PyTorch, etc.)
pip install -r requirements.txt

# Run federated training with GAN attack
python GAN_Data_Leakage.py
```

## References

- Briland Hitaj, Giuseppe Ateniese, and Fernando Perez-Cruz. 2017. *Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning*. In *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS ‘17)*. Association for Computing Machinery, New York, NY, USA, 603–618. [https://doi.org/10.1145/3133956.3134012](https://doi.org/10.1145/3133956.3134012)

- Ligeng Zhu, Zhijian Liu, and Song Han. 2019. *Deep Leakage from Gradients*. arXiv:1906.08935 [cs.LG]. [https://doi.org/10.48550/arXiv.1906.08935](https://doi.org/10.48550/arXiv.1906.08935)

