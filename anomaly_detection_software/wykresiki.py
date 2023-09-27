from matplotlib import pyplot as plt
import os
import pandas as pd
from matplotlib import cm

# wykres pokazujący wpływ wartości mnożnika błędu
# dla jednego błędu

def get_colors_from_labels(labels, colormap_dict):
    colors = []
    for label in labels:
        val = colormap_dict[label].flatten()
        print(val, type(val))
        colors.append(list(val))

def get_parameter(dir_name, parameter=None, sec_param=None, path_ext = None):
    path = "/home/wasyl/magisterka/fault_detection/"
    if path_ext:
        path = path + path_ext
    f1_score = []
    specificity = []
    sensitivity = []
    for item in os.listdir(path):
        for name in dir_name:
            if name in item:
                if os.path.isdir(f"{path}/{item}"):
                    if parameter: 
                        for param in parameter:
                            if sec_param:
                                for par in sec_param:
                                    results = pd.read_csv(f"{path}{item}/{param}/{par}/stats_results_{param}_{par}.csv", header=None)
                                    f1_score.append(results[2][4])
                                    specificity.append(results[2][3])
                                    sensitivity.append(results[2][2])
                            results = pd.read_csv(f"{path}{item}/{param}/stats_results_{param}.csv", header=None)
                            f1_score.append(results[2][4])
                            specificity.append(results[2][3])
                            sensitivity.append(results[2][2])
                    else:
                        results = pd.read_csv(f"{path}/{item}/stats_results.csv", header=None)
                        f1_score.append(results[2][4])
                        specificity.append(results[2][3])
                        sensitivity.append(results[2][2])
                    
    return sum(f1_score)/len(f1_score), sum(specificity)/len(specificity), sum(sensitivity)/len(sensitivity)

# wykres pokazujący wpływ wartości mnożnika błędu

y1_f1, y1_sp, y1_sn = get_parameter(["exp6"])

y2_f1, y2_sp, y2_sn = get_parameter(["exp2"])

y3_f1, y3_sp, y3_sn = get_parameter(["exp3"])

print(y1_f1, y1_sp, y1_sn, y3_f1, y3_sp, y3_sn)

fig, ax = plt.subplots(1,1)
ax.bar([1.8, 2, 2.2], [y1_sp, y1_sn, y1_f1], width=0.16,
        color=['blue', 'green', 'red'], label=["specificity", 'sensivity', 'F1'])
ax.bar([2.8, 3, 3.2], [y2_sp, y2_sn, y2_f1], width=0.16,
        color=['blue', 'green', 'red'])
ax.bar([3.8, 4, 4.2], [y3_sp, y3_sn, y3_f1], width=0.16,
        color=['blue', 'green', 'red'])
# plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
ax.set_xticks([2, 3, 4])
ax.set_xticklabels(["1", "2", "3"], fontsize=12)
plt.ylim(0.2, 1)
plt.xlabel("Wartość mnożnika")
plt.ylabel("Wartość metryki")
plt.legend(loc="lower right")
plt.show()

# # wykres pokazujący wpływ wartości mnożnika błędu dla ml i klastrowania

# y1_f1_ml, _, _ = get_parameter(["exp6"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y1_f1_cl, _, _ = get_parameter(["exp6"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y2_f1_ml, _, _ = get_parameter(["exp2"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y2_f1_cl, _, _ = get_parameter(["exp2"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y3_f1_ml, _, _ = get_parameter(["exp3"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y3_f1_cl, _, _ = get_parameter(["exp3"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# # print(y1_f1, y1_sp, y1_sn, y2_f1, y2_sp, y2_sn, y3_f1, y3_sp, y3_sn)

# fig, ax = plt.subplots(1,1)
# ax.bar([1.8, 2.2], [y1_f1_cl, y1_f1_ml], width=0.36,
#         color=['blue', 'red'], label=["klasteryzacja", "uczenie maszynowe"])
# ax.bar([2.8, 3.2], [y2_f1_cl, y2_f1_ml], width=0.36,
#         color=['blue', 'red'])
# ax.bar([3.8, 4.2], [y3_f1_cl, y3_f1_ml], width=0.36,
#         color=['blue', 'red'])
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax.set_xticks([2, 3, 4])
# ax.set_xticklabels(["1", "2", "3"], fontsize=12)
# plt.ylim(0.5, 1)
# plt.xlabel("Wartość mnożnika")
# plt.ylabel("F1 score")
# plt.legend(loc="upper left")
# plt.show()

# # wykres pokazujący wykrywalność błędów w zależności od mnożnika

# # mnożnik1 
# drift1, _, _ = get_parameter(['exp6'], ["drift"])

# spike1, _, _ = get_parameter(['exp6'], ["spike"])

# hardover1, _, _ = get_parameter(['exp6'], ["hardover"])

# erratic1, _, _ = get_parameter(['exp6'], ["erratic"])


# # mnożnik2
# drift2, _, _ = get_parameter(['exp2'], ["drift"])

# spike2, _, _ = get_parameter(['exp2'], ["spike"])

# hardover2, _, _ = get_parameter(['exp2'], ["hardover"])

# erratic2, _, _ = get_parameter(['exp2'], ["erratic"])

# # mnożnik3
# drift3, _, _ = get_parameter(['exp3'], ["drift"])

# spike3, _, _ = get_parameter(['exp3'], ["spike"])

# hardover3, _, _ = get_parameter(['exp3'], ["hardover"])

# erratic3, _, _ = get_parameter(['exp3'], ["erratic"])

# # print(f1_gm, f1_DB, f1_if, f1_lof, f1_knn, f1_dt, f1_lg, f1_svc)

# data = [[drift1, erratic1, hardover1, spike1], 
#         [drift2, erratic2, hardover2, spike2], 
#         [drift3, erratic3, hardover3, spike3]]
# fig, axes = plt.subplots(1,3)
# for ax, param in zip(axes.flat,data):
#     ax.bar([1, 2, 3, 4], param)
#     # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
#     # ax.set_xticklabels(["erratic", "drift", "spike",  "hardover"],
#     #                 rotation=70, fontsize=8)
#     ax.set_xticks([1, 2, 3, 4])
#     ax.set_xticklabels(["drift", "erratic", "hardover", "spike"], rotation=45, fontsize=8)
#     ax.set(ylabel="F1 score", ylim=(0.4, 1))
#     plt.legend(loc="lower right")
# axes[0].set_title("Mnożnik = 1", fontsize=8)
# axes[1].set_title("Mnożnik = 2", fontsize=8)
# axes[2].set_title("Mnożnik = 3", fontsize=8)
# fig.text(0.5, 0.01, 'Rodzaj błędu', ha='center')
# plt.tight_layout()
# plt.show()

# # wykres pokazujący wpływ ilości dodanych błędów

# y1_f1, y1_sp, y1_sn = get_parameter(["1error"])

# y2_f1, y2_sp, y2_sn = get_parameter(["2errors_same_time"])

# y3_f1, y3_sp, y3_sn = get_parameter(["3errors_same_time"])

# print(y1_f1, y1_sp, y1_sn, y2_f1, y2_sp, y2_sn, y3_f1, y3_sp, y3_sn)

# fig, ax = plt.subplots(1,1)
# ax.bar([1.8, 2, 2.2], [y1_sp, y1_sn, y1_f1], width=0.16,
#         color=['blue', 'green', 'red'], label=["specificity", 'sensivity', 'F1'])
# ax.bar([2.8, 3, 3.2], [y2_sp, y2_sn, y2_f1], width=0.16,
#         color=['blue', 'green', 'red'])
# ax.bar([3.8, 4, 4.2], [y3_sp, y3_sn, y3_f1], width=0.16,
#         color=['blue', 'green', 'red'])
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax.set_xticks([2, 3, 4])
# ax.set_xticklabels(["1", "2", "3"], fontsize=12)
# plt.ylim(0.2, 1)
# plt.xlabel("Ilość dodanych błędów")
# plt.ylabel("Wartość metryki")
# plt.legend(loc="upper left")
# plt.show()

# # wykres pokazujący wpływ parametru same_time

# y2_f1_same, _, _ = get_parameter(["2errors_same_time"])

# y2_f1_diff, _, _  = get_parameter(["2errors_diff_time"])

# y3_f1_same, _, _  = get_parameter(["3errors_same_time"])

# y3_f1_diff, _, _  = get_parameter(["3errors_diff_time"])

# print(y2_f1_same, y2_f1_diff, y3_f1_same, y3_f1_diff)

# fig, ax = plt.subplots(1,1)
# ax.bar([1.8, 2.2], [y2_f1_same, y2_f1_diff], width=0.36,
#         color=['blue', 'red'], label=["błędy w tej samej chwili czasu", "błędy w losowych chwilach czasu"])
# ax.bar([2.8, 3.2], [y3_f1_same, y3_f1_diff], width=0.36,
#         color=['blue', 'red'])

# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax.set_xticks([2, 3])
# ax.set_xticklabels(["2", "3"], fontsize=12)
# plt.ylim(0.2, 1)
# plt.xlabel("Ilość dodanych błędów")
# plt.ylabel("F1 score")
# plt.legend(loc="upper left")
# plt.show()

# # wykres pokazujący wpływ parametru same_time dla klasteryzacji and ML

# y2_f1_same_ml, _, _ = get_parameter(["2errors_same_time"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y2_f1_same_cl, _, _  = get_parameter(["2errors_same_time"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y2_f1_diff_ml, _, _ = get_parameter(["2errors_diff_time"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y2_f1_diff_cl, _, _  = get_parameter(["2errors_diff_time"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y3_f1_same_ml, _, _ = get_parameter(["3errors_same_time"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y3_f1_same_cl, _, _  = get_parameter(["3errors_same_time"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y3_f1_diff_ml, _, _ = get_parameter(["3errors_diff_time"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y3_f1_diff_cl, _, _  = get_parameter(["3errors_diff_time"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# # print(y1_f1, y1_sp, y2_f1, y2_sp, y2_sn, y2_f1, y3_f1, y3_sp, y3_sn)

# fig, ax = plt.subplots(1,2)
# ax[0].bar([1.8, 2.2], [y2_f1_same_ml, y2_f1_same_cl], width=0.36,
#         color=['blue', 'red'], label=["uczenie maszynowe", "klasteryzacja"])
# ax[0].bar([2.8, 3.2], [y3_f1_same_ml, y3_f1_same_cl], width=0.36,
#         color=['blue', 'red'])
# ax[1].bar([1.8, 2.2], [y2_f1_diff_ml, y2_f1_diff_cl], width=0.36,
#         color=['blue', 'red'], label=["uczenie maszynowe", "klasteryzacja"])
# ax[1].bar([2.8, 3.2], [y3_f1_diff_ml, y3_f1_diff_cl], width=0.36,
#         color=['blue', 'red'])

# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax[0].set_xticks([2, 3])
# ax[0].set_xticklabels(["2", "3"], fontsize=8)
# ax[0].set_title("Błędy występujące w tych samych chwilach czasu", fontsize=8)
# ax[1].set_xticks([2, 3])
# ax[1].set_xticklabels(["2", "3"], fontsize=8)
# ax[1].set_title("Błędy występujące w losowych chwilach czasu", fontsize=8)
# # ax[0].ylim(0.2, 1)
# ax[0].set(xlabel="Ilość dodanych błędów", ylabel="F1 score", ylim=(0.6, 1))
# ax[1].set(xlabel="Ilość dodanych błędów", ylabel="F1 score", ylim=(0.6, 1))
# # ax[0].ylabel("F1 score")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.show()

# # wykres pokazujący wpływ ilości dodanych błędów względem ml i klasteryzacji

# y1_f1_ml, _, _ = get_parameter(["1error"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y1_f1_cl, _, _  = get_parameter(["1error"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y2_f1_ml, _, _ = get_parameter(["2errors_same_time"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y2_f1_cl, _, _  = get_parameter(["2errors_same_time"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# y3_f1_ml, _, _ = get_parameter(["3errors_same_time"], ["KNN", "DecisionTree", "SVC", "LogisticRegression"])

# y3_f1_cl, _, _  = get_parameter(["3errors_same_time"], ["DBscan", "Gaussian mixture", "Local outlier factor", "Isolation forests"])

# print(y1_f1_ml, y2_f1_ml, y3_f1_ml, y1_f1_cl, y2_f1_cl, y3_f1_cl)

# fig, ax = plt.subplots(1,1)
# ax.bar([1.8, 2.2], [y1_f1_cl, y1_f1_ml], width=0.36,
#         color=['blue', 'red'], label=["klasteryzacja", "uczenie maszynowe"])
# ax.bar([2.8, 3.2], [y2_f1_cl, y2_f1_ml], width=0.36,
#         color=['blue', 'red'])
# ax.bar([3.8, 4.2], [y3_f1_cl, y3_f1_ml], width=0.36,
#         color=['blue', 'red'])
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax.set_xticks([2, 3, 4])
# ax.set_xticklabels(["1", "2", "3"], fontsize=12)
# plt.ylim(0.6, 1)
# plt.xlabel("Ilość dodanych błędów")
# plt.ylabel("Wartość metryki")
# plt.legend(loc="lower right")
# plt.show()

# # wykres pokazujący średnią wydajność każdego modelu

# colormap = cm.get_cmap('viridis', 8).colors

# models = ["Gaussian mixture", "DBscan", "Isolation forests", 
#           "Local outlier factor", "LogisticRegression", "KNN", "SVC", "DecisionTree"]

# color_dict = dict(zip(models, colormap))
# # print(color_dict)

# phrase = ["3errors"]

# f1_knn, sp_knn, _ = get_parameter(phrase, ["KNN"])

# f1_dt, sp_dt, sn_dt = get_parameter(phrase, ["DecisionTree"])

# f1_lg, sp_lg, _ = get_parameter(phrase, ["LogisticRegression"])

# f1_svc, sp_svc, _ = get_parameter(phrase, ["SVC"])

# f1_DB, sp_DB, _ = get_parameter(phrase, ["DBscan"])

# f1_gm, sp_gm, _ = get_parameter(phrase, ["Gaussian mixture"])

# f1_lof, sp_lof, _ = get_parameter(phrase, ["Local outlier factor"])

# f1_if, sp_if, sn_if = get_parameter(phrase, ["Isolation forests"])

# print(f1_dt, sp_dt, sn_dt, f1_if, sp_if, sn_if)

# fig, ax = plt.subplots(1,2)
# ax[0].bar([1, 2, 3, 4, 5, 6, 7, 8], [f1_gm, f1_DB, f1_if, f1_lof, f1_lg, f1_knn, f1_svc, f1_dt], 
#           color=get_colors_from_labels(["Gaussian mixture", "DBscan", "Isolation forests", "Local outlier factor",
#                     "LogisticRegression", "KNN", "SVC", "DecisionTree"], color_dict))
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax[0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
# ax[0].set_xticklabels(["Gaussian mixture", "DBscan", "Isolation forest", "Local outlier factor",
#                     "LogisticRegression", "KNN", "SVC", "DecisionTree"],
#                    rotation=70, fontsize=8)
# ax[1].bar([1, 2, 3, 4, 5, 6, 7, 8], [sp_gm, sp_DB, sp_if, sp_lof, sp_lg, sp_knn, sp_svc, sp_dt],
#           color=get_colors_from_labels(["Gaussian mixture", "DBscan", "Isolation forests", "Local outlier factor",
#                     "LogisticRegression", "KNN", "SVC", "DecisionTree"], color_dict))
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax[1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
# ax[1].set_xticklabels(["Gaussian mixture", "DBscan", "Isolation forest", "Local outlier factor",
#                     "LogisticRegression", "KNN", "SVC", "DecisionTree"],
#                    rotation=70, fontsize=8)
# # ax[0].set_title("Mnożnik = 1", fontsize=8)
# # ax[1].set_title("Mnożnik = 2", fontsize=8)
# ax[0].set(ylabel="F1 score", ylim=(0.0, 1))
# ax[1].set(ylabel="Specificity", ylim=(0.0, 1))
# fig.text(0.5, 0.01, 'Nazwa modelu', ha='center')
# plt.tight_layout()
# plt.show()

# # wykres pokazujący dokladne rezultaty decision tree
# model1 = "Isolation forests"
# model2 = "DecisionTree"
# # wpływ mnożnika
# m1_multiplier1_f1, m1_multiplier1_sp, m1_multiplier1_sn = get_parameter(['exp6'], parameter=[model1])

# m1_multiplier2_f1, m1_multiplier2_sp, m1_multiplier2_sn = get_parameter(['exp2'], parameter=[model1])

# m1_multiplier3_f1, m1_multiplier3_sp, m1_multiplier3_sn = get_parameter(['exp3'], parameter=[model1])

# m2_multiplier1_f1, m2_multiplier1_sp, m2_multiplier1_sn = get_parameter(['exp6'], parameter=[model2])

# m2_multiplier2_f1, m2_multiplier2_sp, m2_multiplier2_sn = get_parameter(['exp2'], parameter=[model2])

# m2_multiplier3_f1, m2_multiplier3_sp, m2_multiplier3_sn = get_parameter(['exp3'], parameter=[model2])


# # wpływ rodziaju błedu
# m1_drift_f1, m1_drift_sp, m1_drift_sn = get_parameter(['exp'], parameter=["drift"], sec_param=[model1])

# m1_spike_f1, m1_spike_sp, m1_spike_sn = get_parameter(['exp'], parameter=["spike"], sec_param=[model1])

# m1_hardover_f1, m1_hardover_sp, m1_hardover_sn = get_parameter(['exp'], parameter=["hardover"], sec_param=[model1])

# m1_erratic_f1, m1_erratic_sp, m1_erratic_sn = get_parameter(['exp'], parameter=["erratic"], sec_param=[model1])

# m2_drift_f1, m2_drift_sp, m2_drift_sn = get_parameter(['exp'], parameter=["drift"], sec_param=[model2])

# m2_spike_f1, m2_spike_sp, m2_spike_sn = get_parameter(['exp'], parameter=["spike"], sec_param=[model2])

# m2_hardover_f1, m2_hardover_sp, m2_hardover_sn = get_parameter(['exp'], parameter=["hardover"], sec_param=[model2])

# m2_erratic_f1, m2_erratic_sp, m2_erratic_sn = get_parameter(['exp'], parameter=["erratic"], sec_param=[model2])

# # wpływ ilości kolumn
# m1_error1_f1, m1_error1_sp, m1_error1_sn = get_parameter(['1error'], [model1])

# m1_error2_f1, m1_error2_sp, m1_error2_sn = get_parameter(['2errors'], [model1])

# m1_error3_f1, m1_error3_sp, m1_error3_sn = get_parameter(['3errors'], [model1])

# m2_error1_f1, m2_error1_sp, m2_error1_sn = get_parameter(['1error'], [model2])

# m2_error2_f1, m2_error2_sp, m2_error2_sn = get_parameter(['2errors'], [model2])

# m2_error3_f1, m2_error3_sp, m2_error3_sn = get_parameter(['3errors'], [model2])


# # print(f1_gm, f1_DB, f1_if, f1_lof, f1_knn, f1_dt, f1_lg, f1_svc)

# fig, axes = plt.subplots(1,1)
# axes.bar([0.7, 0.8, 0.9], 
#             [m1_multiplier1_f1, m1_multiplier1_sp, m1_multiplier1_sn],
#             color=['blue', 'green', 'red'], 
#             label=["f1 score", "specificity", "sensitivity"], width=0.08)
# axes.bar([1.1, 1.2, 1.3, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3], 
#             [m2_multiplier1_f1, m2_multiplier1_sp, m2_multiplier1_sn,
#              m1_multiplier2_f1, m1_multiplier2_sp, m1_multiplier2_sn,
#              m2_multiplier2_f1, m2_multiplier2_sp, m2_multiplier2_sn,
#              m1_multiplier3_f1, m1_multiplier3_sp, m1_multiplier3_sn,
#              m2_multiplier3_f1, m2_multiplier3_sp, m2_multiplier3_sn],
#             color=['blue', 'green', 'red'], width=0.08)
# axes.set_xticks([0.8, 1, 1.2, 1.8, 2, 2.2, 2.8, 3.0, 3.2])
# axes.set_xticklabels(["las izolacji", "1", "drzewa decyzyjne", "las izolacji", "2", "drzewa decyzyjne", "las izolacji", "3", "drzewa decyzyjne"], 
#                      rotation=60, fontsize=8)
# axes.set(xlabel="wielkosc mnożnika", ylabel="Wartość metryki", ylim=(0.0, 1))
# axes.set_title("Skuteczność modeli w zależności od mnożnika", fontsize=8)
# plt.legend(loc="lower right")
# # fig.text(0.5, 0.01, 'Rodzaj błędu', ha='center')
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1,1)
# axes.bar([0.7, 0.8, 0.9], 
#             [m1_error1_f1, m1_error1_sp, m1_error1_sn],
#             color=['blue', 'green', 'red'], 
#             label=["f1 score", "specificity", "sensitivity"], width=0.08)
# axes.bar([1.1, 1.2, 1.3, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3], 
#             [m2_error1_f1, m2_error1_sp, m2_error1_sn,
#              m1_error2_f1, m1_error2_sp, m1_error2_sn,
#              m2_error2_f1, m2_error2_sp, m2_error2_sn,
#              m1_error3_f1, m1_error3_sp, m1_error3_sn,
#              m2_error3_f1, m2_error3_sp, m2_error3_sn],
#             color=['blue', 'green', 'red'], width=0.08)
# axes.set_xticks([0.8, 1, 1.2, 1.8, 2, 2.2, 2.8, 3.0, 3.2])
# axes.set_xticklabels(["las izolacji", "1", "drzewa decyzyjne", "las izolacji", "2", "drzewa decyzyjne", "las izolacji", "3", "drzewa decyzyjne"], 
#                      rotation=60, fontsize=8)
# axes.set(xlabel="Ilość dodanych błędów", ylabel="Wartość metryki", ylim=(0.0, 1))
# axes.set_title("Skuteczność modeli w zależności od ilości zakłóconych czujników", fontsize=8)
# plt.legend(loc="lower right")
# # fig.text(0.5, 0.01, 'Rodzaj błędu', ha='center')
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1,1)
# axes.bar([0.7, 0.8, 0.9], 
#             [m1_erratic_f1, m1_erratic_sp, m1_erratic_sn],
#             color=['blue', 'green', 'red'], 
#             label=["f1 score", "specificity", "sensitivity"], width=0.08)
# axes.bar([1.1, 1.2, 1.3, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3], 
#             [m2_erratic_f1, m2_erratic_sp, m2_erratic_sn,
#              m1_hardover_f1, m1_hardover_sp, m1_hardover_sn,
#              m2_hardover_f1, m2_hardover_sp, m2_hardover_sn,
#              m1_spike_f1, m1_spike_sp, m1_spike_sn,
#              m2_spike_f1, m2_spike_sp, m2_spike_sn,
#              m1_drift_f1, m1_drift_sp, m1_drift_sn,
#              m2_drift_f1, m2_drift_sp, m2_drift_sn],
#             color=['blue', 'green', 'red'], width=0.08)
# axes.set_xticks([0.8, 1, 1.2, 1.8, 2, 2.2, 2.8, 3.0, 3.2, 3.8, 4, 4.2])
# axes.set_xticklabels(["las izolacji", "erratic", "drzewa decyzyjne", "las izolacji", "hardover", "drzewa decyzyjne", 
#                       "las izolacji", "spike", "drzewa decyzyjne", "las izolacji", "drift", "drzewa decyzyjne"], 
#                      rotation=60, fontsize=8)
# axes.set(xlabel="Rodzaj błędu", ylabel="Wartość metryki", ylim=(0.0, 1))
# axes.set_title("Wykrywalność poszczególnych rodzajów błędów", fontsize=8)
# plt.legend(loc="lower right")
# # fig.text(0.5, 0.01, 'Rodzaj błędu', ha='center')
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(1,1)
# axes[0].bar([0.8, 1, 1.2, 1.8, 2, 2.2, 2.8, 3, 3.2, 3.8, 4, 4.2], 
#             [drift_f1, drift_sp, drift_sn,
#              spike_f1, spike_sp, spike_sn,
#              hardover_f1, hardover_sp, hardover_sn,
#              erratic_f1, erratic_sp, erratic_sn],
#             color=['blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'], 
#             label=["f1", "sp", "sn", "f1", "sp", "sn", "f1", "sp", "sn", "f1", "sp", "sn"], width=0.16)
# axes[0].set_xticks([1, 2, 3, 4])
# axes[0].set_xticklabels(["drift", "spike", "hardover", "erratic"], rotation=45, fontsize=8)
# axes[0].set(xlabel="Rozdaj bledu", ylabel="Wartość metryki", ylim=(0.0, 1))
# axes[0].set_title("Wykrywalnosć błędów", fontsize=8)

# fig, axes = plt.subplots(1,1)
# axes[0].bar([0.8, 1, 1.2, 1.8, 2, 2.2, 2.8, 3, 3.2], 
#             [error1_f1, error1_sp, error1_sn,
#              error2_f1, error2_sp, error2_sn,
#              error3_f1, error3_sp, error3_sn],
#             color=['blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'], 
#             label=["f1", "sp", "sn", "f1", "sp", "sn", "f1", "sp", "sn"], width=0.16)
# axes[0].set_xticks([1, 2, 3])
# axes[0].set_xticklabels(["1", "2", "3"], fontsize=8)
# axes[0].set(xlabel="Ilość dodanych błędów", ylabel="Wartość metryki", ylim=(0.0, 1))
# axes[0].set_title("Wpływ błędów", fontsize=8)

# plt.legend(loc="lower right")
# # fig.text(0.5, 0.01, 'Rodzaj błędu', ha='center')
# # plt.tight_layout()
# plt.show()


# # DOMAIN ADAPTATIONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# # domain adaptation na biorze KIA

# y1_f1, y1_sp, y1_sn = get_parameter(["exp_da"], path_ext="results_domain_adaptation/")

# y2_f1, y2_sp, y2_sn = get_parameter(["exp_no_da"], path_ext="results_domain_adaptation/")

# y3_f1, y3_sp, y3_sn = get_parameter(["exp2"], ["kia_soul"])

# # print(y1_f1, y1_sp, y1_sn, y2_f1, y2_sp, y2_sn, y3_f1, y3_sp, y3_sn)

# fig, ax = plt.subplots(1,1)
# ax.bar([1.8, 2, 2.2], [y1_sp, y1_sn, y1_f1], width=0.16,
#         color=['blue', 'green', 'red'], label=["specificity", 'sensivity', 'F1'])
# ax.bar([2.8, 3, 3.2], [y2_sp, y2_sn, y2_f1], width=0.16,
#         color=['blue', 'green', 'red'])
# ax.bar([3.8, 4, 4.2], [y3_sp, y3_sn, y3_f1], width=0.16,
#         color=['blue', 'green', 'red'])
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax.set_xticks([2, 3, 4])
# ax.set_xticklabels(["adaptacja domen", "połączone pliki", "sredni wynik"], fontsize=12)
# plt.ylim(0.2, 1)
# plt.xlabel("Wartość mnożnika")
# plt.ylabel("Wartość metryki")
# plt.legend(loc="upper left")
# plt.show()

# # domain adaptation na biorze KIA

# y1_f1, y1_sp, y1_sn = get_parameter(["exp2_da"], path_ext="results_domain_adaptation/")

# y2_f1, y2_sp, y2_sn = get_parameter(["exp2_no_da"], path_ext="results_domain_adaptation/")

# y3_f1, y3_sp, y3_sn = get_parameter(["exp2_da_control_group"], path_ext="results_domain_adaptation/")

# print(y1_f1, y1_sp, y1_sn, y2_f1, y2_sp, y2_sn, y3_f1, y3_sp, y3_sn)

# fig, ax = plt.subplots(1,1)
# ax.bar([1.8, 2, 2.2], [y1_sp, y1_sn, y1_f1], width=0.16,
#         color=['blue', 'green', 'red'], label=["specificity", 'sensivity', 'F1'])
# ax.bar([2.8, 3, 3.2], [y2_sp, y2_sn, y2_f1], width=0.16,
#         color=['blue', 'green', 'red'])
# ax.bar([3.8, 4, 4.2], [y3_sp, y3_sn, y3_f1], width=0.16,
#         color=['blue', 'green', 'red'])
# # plt.bar([1, 2, 3], [y1_f1, y2_f1, y3_f1])
# ax.set_xticks([2, 3, 4])
# ax.set_xticklabels(["adaptacja domen", "połączone pliki", "sredni wynik"], fontsize=12)
# plt.ylim(0.2, 1)
# plt.xlabel("Wartość mnożnika")
# plt.ylabel("Wartość metryki")
# plt.legend(loc="upper left")
# plt.show()