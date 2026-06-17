"""Data-fetching package — market prices, FRED macro series, ETF fund flows.

All functions are decorated with @st.cache_data; cache behavior is unchanged by
living in this package. Import via `from data.market import fetch_macro_data`, etc.
"""
