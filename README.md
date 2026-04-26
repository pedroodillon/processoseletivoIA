# Classificação de Dígitos com Edge AI (MNIST)

## 👤 Identificação
Nome: Pedro Odillon F. M. M. Figueiredo  
GitHub: pedroodillon 

---

## 1️⃣ Resumo da Arquitetura do Modelo

Foi implementada uma Rede Neural Convolucional (CNN) para classificação de dígitos do dataset MNIST.

A arquitetura utilizada foi:

```text
Entrada (28x28x1)
↓
Conv2D (32 filtros) + ReLU
↓
MaxPooling2D
↓
Conv2D (64 filtros) + ReLU
↓
MaxPooling2D
↓
Flatten
↓
Dense (64) + ReLU
↓
Dense (10) + Softmax
```

As camadas convolucionais extraem padrões visuais das imagens (como bordas e formas), enquanto o MaxPooling reduz a dimensionalidade e o custo computacional.

A camada Dense realiza a classificação final, e a função Softmax fornece a probabilidade de cada classe.

A arquitetura foi mantida simples de forma intencional, priorizando eficiência computacional e compatibilidade com execução em ambientes de Edge AI.

---

## 2️⃣ Bibliotecas Utilizadas

- TensorFlow / Keras  
- NumPy  

---

## 3️⃣ Técnica de Otimização do Modelo

O modelo treinado foi convertido para o formato TensorFlow Lite (`.tflite`) com aplicação de **Dynamic Range Quantization**.

Essa técnica reduz o tamanho do modelo ao diminuir a precisão dos pesos durante a inferência, resultando em menor consumo de memória e melhor eficiência em dispositivos embarcados.

A escolha dessa técnica foi feita por oferecer um bom equilíbrio entre simplicidade, compatibilidade e redução de tamanho.

Como melhoria futura, poderia ser aplicada a **quantização inteira completa (Full Integer Quantization)**, que tende a reduzir ainda mais o custo computacional, porém exige um conjunto representativo de dados para calibração.

---

## 4️⃣ Resultados Obtidos

O modelo atingiu aproximadamente **98.9% de acurácia** no conjunto de teste, com **loss aproximada de 0.03**, demonstrando bom desempenho mesmo com uma arquitetura leve.

Comparação de tamanho dos modelos:

| Modelo | Formato | Tamanho |
|--------|--------|---------|
| Modelo treinado | `.h5` | ~1467 KB |
| Modelo otimizado | `.tflite` | ~128 KB |

A redução de tamanho foi significativa (~90%), mantendo o desempenho praticamente inalterado.

Esse resultado demonstra que o modelo é adequado para aplicações em Edge AI, onde há restrições de memória e processamento.

---

## 5️⃣ Comentários Adicionais

### Contexto de aplicação

Este projeto foi desenvolvido considerando cenários de Edge AI e Indústria 4.0, onde modelos de Machine Learning são executados diretamente em dispositivos embarcados.

Em aplicações reais, esse tipo de solução pode ser utilizado em sistemas de inspeção visual, automação industrial e dispositivos inteligentes, onde decisões precisam ser tomadas localmente e em tempo real.

---

### Decisões técnicas

- Utilização de uma CNN simples para reduzir custo computacional  
- Limitação do número de épocas para permitir execução em ambiente de CI/CD  
- Separação do pipeline em etapas (treinamento e otimização)  
- Aplicação de quantização para reduzir o tamanho do modelo  

A principal decisão foi priorizar o equilíbrio entre **desempenho, tamanho do modelo e custo computacional**, ao invés de maximizar apenas a acurácia.

---

### Limitações

- O modelo foi treinado em um dataset simples (MNIST)  
- Não foram exploradas arquiteturas mais complexas  
- A otimização utilizou apenas quantização dinâmica  

---

### Aprendizados

- Construção de um pipeline completo de Machine Learning  
- Otimização de modelos para execução em dispositivos de borda  
- Uso de GitHub Actions para validação automatizada  
- Importância de considerar restrições de hardware no desenvolvimento de modelos  

---

### Conexão com IoT

Este projeto representa a etapa de **percepção** em um sistema de Edge AI.

Em uma aplicação real, o modelo otimizado poderia ser embarcado em dispositivos como o ESP32-S3 CAM, onde a inferência é realizada localmente.

Os resultados da classificação podem ser utilizados por sistemas embarcados (IoT) para acionar atuadores, sinalizar estados ou tomar decisões em tempo real, conectando a inteligência do modelo ao mundo físico.