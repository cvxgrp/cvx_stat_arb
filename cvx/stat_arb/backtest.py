# import pandas as pd
# from cvx.stat_arb.ccp import construct_stat_arb



# def filter_stat_arbs(all_stat_arbs, prices_train, prices_val, prices_test,\
#      cutoff=1.05, SR_cutoff=3):

#     prices_train_val = pd.concat([prices_train, prices_val])

#     cutoff = 1.05

#     total_profit = 0

#     SR_cutoff = 3

#     traded_stat_arbs = []
#     non_traded_stat_arbs = []

#     for i, stat_arb in enumerate(all_stat_arbs):

#         # Consider stat arb for trading if it stays within cutoff
#         if (stat_arb.evaluate(prices_val)-stat_arb.mu).abs().max()<=cutoff:

#             metrics_train = stat_arb.metrics(prices_train, cutoff=cutoff)
#             metrics_val = stat_arb.metrics(prices_val, cutoff=cutoff)

#             # Consider stat arb for trading if it has high SR on validation data
#             if metrics_train.sr_profit>SR_cutoff and\
#                 metrics_val.sr_profit is not None\
#                 and metrics_val.sr_profit>SR_cutoff:
            
#                 # Refit with validation data
#                 p_init = stat_arb.evaluate(prices_train_val).values.reshape(-1,1)
#                 P = prices_train_val[stat_arb.assets.keys()]
#                 stat_arb_refit = construct_stat_arb(P, P_max=None,\
#                         s_init=stat_arb.s, mu_init=stat_arb.mu)

#                 # Don't enter position if it goes out of bounds on first day
#                 if (stat_arb_refit.evaluate(prices_test)-stat_arb_refit.mu).abs()[0]\
#                     >= cutoff: 
#                     non_traded_stat_arbs.append(stat_arb_refit) 
#                 else:
#                     # print(1)
#                     metrics = stat_arb_refit.metrics(prices_test, cutoff=cutoff)
#                     total_profit += metrics.total_profit

#                     # Append to traded stat arb list
#                     traded_stat_arbs.append(stat_arb_refit)

#             else:
#                 # Append to non-traded stat arb list
#                 non_traded_stat_arbs.append(stat_arb)

#     return traded_stat_arbs, non_traded_stat_arbs, total_profit