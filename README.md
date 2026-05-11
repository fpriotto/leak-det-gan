# Detecção e Localização de Vazamentos com WGAN-GP e ViT

Repositório com o código-fonte da dissertação *"Modelos Generativos Aplicados à Emissão Acústica para Detecção e Localização de Vazamentos em Tubulações"*.

Este projeto utiliza Redes Adversárias Generativas (WGAN-GP Dual-Critic) para realizar *Data Augmentation* de sinais de Emissão Acústica (EA). Os sinais sintéticos e reais são convertidos em espectrogramas logarítmicos e avaliados sob diferentes níveis de ruído AWGN utilizando modelos *Vision Transformer (ViT)* e CNNs (ResNet-18) em cascata.

---

## 📌 Visão Geral dos Experimentos

O pipeline de testes deste repositório foi construído para avaliar a robustez da geração de dados em cenários limitados (50% dos dados de treino) e submeter os classificadores a testes de estresse com ruído térmico (SNR).

Os experimentos principais incluem:
1. **Stress Test de Ruído (AWGN):** Avaliação dos modelos ViT treinados a 50dB nos cenários de 40, 30, 20 e 10 dB.
2. **Ablação de Resolução:** Comparativo do *Vision Transformer* utilizando espectrogramas em 32x32, 64x64 e 128x128 pixels.
3. **Ablação Arquitetural (Classificadores):** Substituição do modelo ViT por uma CNN Simples e uma ResNet-18.
4. **Ablação Generativa:** Comparativo da WGAN-GP (com crítico temporal e espectral) contra uma Vanilla CGAN básica.

---

## ⚙️ Estrutura do Projeto e Pipeline

Todos os scripts devem ser executados a partir da raiz do projeto.

* `scripts/00_convert_npy_to_pkl.py`: Descompacta os dumps raw de memória zlib (1MHz) em `.pkl`.
* `scripts/01_prepare_data.py`: Fatiamento (10ms) e aplicação de filtros Butterworth e Chebyshev.
* `scripts/02_train_wgan.py`: Treinamento do modelo generativo (WGAN-GP ou CGAN).
* `scripts/03_generate_specs.py`: Geração dos espectrogramas com injeção de ruído (AWGN) em múltiplos SNRs.
* `scripts/04_train_evaluate_vit.py`: Treina os modelos ViT/ResNet a 50dB e avalia o desempenho em dados reais (10 a 50dB).
* `scripts/05_model_complexity.py`: Calcula FLOPs (MACs) e número de parâmetros dos modelos para os relatórios.

---

## 🚀 Como reproduzir os experimentos

Antes de iniciar, instale as dependências:
```bash
pip install -r requirements.txt
pip install thop  # Necessário para calcular FLOPs