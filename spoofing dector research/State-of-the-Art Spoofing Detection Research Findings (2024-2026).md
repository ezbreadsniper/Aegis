# State-of-the-Art Spoofing Detection Research Findings (2024-2026)

## 1. "Learning the Spoofability of Limit Order Books" (Fabre & Challet, 2025)

This is the most advanced, real-time spoofing detection framework for cryptocurrency exchanges.

### Key Innovations
- **Multi-Scale Hawkes Features**: Novel order flow variables based on Hawkes processes that account for both the SIZE and PLACEMENT DISTANCE of limit orders.
- **Probabilistic Neural Network**: A neural network trained to predict the conditional probability distribution of mid-price movements.
- **Spoofing Gain Framework**: Detection based on the "probabilistic market manipulation gain" of a spoofing agent.

### Critical Finding
> "31% of large orders could spoof the market."

### Key Insight for Our Bot
The paper proves that **posting distance** is critical. Spoofing detection models that do not account for the distance of orders from the best price are inadequate. Our current spoof detector must be upgraded to include this feature.

## 2. "Deep Unsupervised Anomaly Detection in High-Frequency Markets" (Poutré et al., 2024)

This is the state-of-the-art unsupervised framework for LOB anomaly detection.

### Key Innovations
- **Transformer Autoencoder**: A modified Transformer architecture to learn rich temporal LOB subsequence representations.
- **Dissimilarity Function**: A learned function in the representation space to characterize "normal" LOB behavior.
- **Trade-Based Manipulation Simulation**: A methodology to generate synthetic quote stuffing, layering, and pump-and-dump manipulations.

### Key Insight for Our Bot
This framework is **unsupervised**, meaning it does not require labeled fraud data. It can detect novel manipulation patterns that have never been seen before. We should consider integrating a similar autoencoder-based anomaly detection layer.

## 3. "GDet: Leveraging Graph Neural Networks for Intelligent Detection of Trade-Based Market Manipulation" (2025)

This is the first GNN-based framework for trade-based manipulation detection.

### Key Innovations
- **Graph Representation**: Modeling trading relationships as a graph, where nodes are accounts and edges are transactions.
- **GNN Architecture**: Using Graph Neural Networks to capture the relational structure of manipulative trading patterns.

### Key Insight for Our Bot
GNNs can capture "ring patterns" of payments and coordinated trading that are invisible to traditional methods. This is a potential future upgrade path for detecting sophisticated, multi-account manipulation.

## 4. "Detecting Multilevel Manipulation from Limit Order Book via Cascaded Contrastive Representation Learning" (Lin & Yang, 2025)

This is the most advanced representation learning framework for LOB manipulation detection.

### Key Innovations
- **Cascaded LOB Representation**: A multi-level architecture that captures information at different depths of the order book.
- **Supervised Contrastive Learning**: Training the model to pull together representations of similar manipulation types and push apart representations of different types.

### Key Insight for Our Bot
Contrastive learning can create a "manipulation embedding space" where different types of spoofing (layering, flickering, vacuuming) cluster together, enabling more robust detection.
