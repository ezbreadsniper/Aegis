# Event-Driven Arbitrage Findings: News, Sentiment, and Front-Running

## 1. The "News Scalping" Edge
- **Information Asymmetry**: Prediction markets like Polymarket are often slower to react to news than crypto exchanges or social media. A bot that can process news in milliseconds can front-run the price adjustment on Polymarket [1].
- **Post-Announcement Drift**: Prices in prediction markets often exhibit "drift" after an event, where they adjust slowly rather than instantly. This creates a window for bots to enter positions before the full price correction.

## 2. Sentiment-Based Front-Running
- **Whale Sentiment**: Large trades on Polymarket can be driven by "insider" sentiment or forced liquidations. Monitoring whale wallets and their sentiment (via social media or trade flow) allows a bot to copy-trade or front-run their moves.
- **Social Media Scrapers**: Bots use real-time scrapers for X (formerly Twitter), Telegram, and Discord to detect "breaking" sentiment before it hits the news wires.
- **Sentiment Accuracy**: Modern AI models (like LLMs) can achieve >70% accuracy in predicting market direction based on daily sentiment indicators [2].

## 3. Event-Driven Execution Mechanics
- **Event Extraction**: Distinguishing between "noise" and "events" is critical. An event is a specific, verifiable occurrence (e.g., "Fed raises rates by 25bps") that has a direct impact on a market's resolution.
- **Sentiment-to-Probability Mapping**: Converting a sentiment score (e.g., "Highly Bullish") into a probability adjustment for the PPO agent.
- **Flash Events**: During "flash" news events, the bot should switch from Post-Only to **Market Orders** to ensure execution before the price gaps.

## 4. Key Sources for Event-Driven Bots
| Source | Type | Speed |
| :--- | :--- | :--- |
| **X (Twitter)** | Social Sentiment | Ultra-Fast |
| **Bloomberg/Reuters** | Institutional News | Fast |
| **Polymarket Activity** | Whale Trade Flow | Real-Time |
| **Crypto Exchanges** | Price Action | Real-Time |
