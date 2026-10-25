from fetch_data import CFG,current_news
for symbols in CFG['symbols'].values():
 for symbol in symbols:current_news(symbol)
