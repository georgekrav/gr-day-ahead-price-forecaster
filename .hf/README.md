---
title: GR Day-Ahead Electricity Price Forecast
emoji: ⚡
colorFrom: blue
colorTo: yellow
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# GR Day-Ahead Electricity Price Forecast

Daily forecast of the 24 hourly day-ahead electricity prices for the Greek
bidding zone (`10YGR-HTSO-----Y`), issued every morning before the SDAC gate
closure. LightGBM on ENTSO-E data, split-conformal prediction intervals,
live track record.

This Space is a read-only viewer: forecasts are produced by a daily GitHub
Actions job and synced here automatically.

Source code, methodology and backtest:
[github.com/georgekrav/gr-day-ahead-price-forecaster](https://github.com/georgekrav/gr-day-ahead-price-forecaster)
