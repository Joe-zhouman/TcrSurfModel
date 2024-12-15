from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score


def plot_learning_curve(
    estimator,  # 估计器，通常是指模型
    title: str,  # 图表标题
    X,  # 特征矩阵
    y,  # 目标向量
    scoring,  # 评估指标
    dpi=1200,
    cv=10,  # 交叉验证的分割数
    train_sizes=np.linspace(0.1, 1.0, 5),  # 训练集大小的序列
):
    """
    绘制学习曲线。

    该函数用于评估模型在不同大小的训练集上的性能。它会绘制出随着训练集大小的增加，
    模型的训练分数和交叉验证分数的变化情况，以此来评估模型的性能。

    参数:
    - estimator: 用于绘制学习曲线的模型。
    - title: 图形的标题。
    - X: 特征矩阵。
    - y: 目标向量。
    - scoring: 用于模型评估的指标。
    - cv: 交叉验证的分割数，默认为10。
    - train_sizes: 训练集大小的序列，用于指定学习曲线中训练集的不同大小。

    返回:
    - 返回绘制学习曲线的Axes对象。
    """
    plt.figure(dpi=dpi)  # 创建一个新的图形
    plt.title(title)  # 设置图形的标题
    plt.xlabel("Training examples")  # 设置x轴的标签
    plt.ylabel("Score")  # 设置y轴的标签

    # 计算学习曲线，返回训练集大小，训练分数和测试分数
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring=scoring, cv=cv, train_sizes=train_sizes
    )

    # 计算训练分数和测试分数的平均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()  # 添加网格线

    # 绘制训练分数的置信区间
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="g",
    )
    # 绘制测试分数的置信区间
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="r",
    )
    # 绘制训练分数曲线
    plt.plot(train_sizes, train_scores_mean, "o-", color="g", label="Training score")
    # 绘制测试分数曲线
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="r", label="Cross-validation score"
    )
    plt.legend(loc="best")  # 添加图例

    return plt.gca()  # 返回当前的Axes对象


def show_feature_text(ax, feature_name, fontsize=20, facecolor="lightblue"):
    """
    在给定的绘图轴上显示特征名称文本。

    该函数的目的是在matplotlib生成的图中，以特定的样式和位置显示特征名称。
    它通过调整轴的属性来确保特征名称居中显示，并且背景颜色可自定义。

    参数:
    - ax: matplotlib的子图轴对象，用于显示文本。
    - feature_name: 字符串，要显示的特征名称。
    - fontsize: 整数，可选，默认为20。特征名称文本的字体大小。
    - facecolor: 字符串，可选，默认为"lightblue"。子图轴的背景颜色。

    返回值:
    无
    """
    # 关闭轴的刻度显示，以便更好地突出特征名称
    ax.axis("off")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 在子图轴的中央添加文本，自动对齐并应用指定的字体大小和颜色
    ax.text(
        0.5,
        0.5,
        feature_name,
        fontsize=fontsize,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    # 设置子图轴的背景颜色
    ax.set_facecolor(facecolor)


def plot_heatmap_with_scatter(
    df_display,
    labelsize=20,
    dpi=1200,
    feature_kwds={"color": "lightblue", "fontsize": 20},
    hplot_kwds={
        "color": "red",
        "kde": True,
        "element": "step",
        "face_color": "#efd6d7",
    },
    splot_kwds={"color": "green", "face_color": "#e4e9ed"},
    cmap_setted=sns.diverging_palette(240, 10, as_cmap=True),
):
    corr = df_display.corr()
    displayed_feature = list(df_display.columns)
    n = len(displayed_feature)
    wa = [3] * n
    wa.append(1)
    ha = [3] * n
    ha.insert(0, 1)
    fig, axes = plt.subplots(
        n + 1,
        n + 1,
        figsize=(2.5 * n, 2.5 * n),
        dpi=dpi,
        gridspec_kw={
            "width_ratios": wa,
            "height_ratios": ha,
        },
    )

    for j in range(0, n):
        show_feature_text(
            ax=axes[0, j],
            feature_name=displayed_feature[j],
            fontsize=feature_kwds["fontsize"],
            facecolor=feature_kwds["color"],
        )
        show_feature_text(
            ax=axes[j + 1, n],
            feature_name=displayed_feature[j],
            fontsize=feature_kwds["fontsize"],
            facecolor=feature_kwds["color"],
        )

    axes[0, n].axis("off")
    for i in range(1, n + 1):
        for j in range(0, n):
            ax = axes[i, j]
            ax.tick_params(axis="both", labelsize=labelsize)
            if i == j + 1:
                ax.set_facecolor(hplot_kwds["face_color"])
                sns.histplot(
                    df_display.iloc[:, i - 1],
                    kde=hplot_kwds["kde"],
                    color=hplot_kwds["color"],
                    element=hplot_kwds["element"],
                    ax=ax,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
            elif i > j + 1:
                ax.set_facecolor(splot_kwds["face_color"])
                sns.scatterplot(
                    x=df_display.iloc[:, j],
                    y=df_display.iloc[:, i - 1],
                    color=splot_kwds["color"],
                    ax=ax,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
            else:
                sns.heatmap(
                    pd.DataFrame([[corr.iloc[i - 1, j]]]),
                    cmap=cmap_setted,
                    cbar=False,
                    annot=True,
                    fmt=".2f",
                    square=True,
                    ax=ax,
                    vmin=-1,
                    vmax=1,
                    annot_kws={"size": 30},
                )
                ax.axis("off")
            if i < n:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap_setted, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=labelsize)
    return axes


def plot_prediction_result(
    train_pred,
    test_pred,
    y_train,
    y_test,
    model_name,
    data_trans,
    fontsize=20,
    figsize=(8, 6),
    dpi=1200,
    palette={"Train": "#b4d4e1", "Test": "#f4ba8a"},
):
    train_pred = data_trans(train_pred)
    test_pred = data_trans(test_pred)
    y_train = data_trans(y_train)
    y_test = data_trans(y_test)
    dataset_train = pd.DataFrame(
        {"True": y_train, "Predicted": train_pred, "Data Set": "Train"}
    )

    dataset_test = pd.DataFrame(
        {"True": y_test, "Predicted": test_pred, "Data Set": "Test"}
    )

    dataset = pd.concat([dataset_train, dataset_test])

    plt.figure(figsize=figsize, dpi=dpi)
    g = sns.JointGrid(
        data=dataset,
        palette=palette,
        x="True",
        y="Predicted",
        hue="Data Set",
        height=10,
    )
    g.plot_joint(sns.scatterplot, alpha=0.8)

    sns.regplot(
        data=dataset_train,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color=palette["Train"],
        label="Train Regression Line",
    )

    sns.regplot(
        data=dataset_test,
        x="True",
        y="Predicted",
        scatter=False,
        ax=g.ax_joint,
        color=palette["Test"],
        label="Test Regression Line",
    )
    g.plot_marginals(
        sns.histplot, kde=True, element="bars", multiple="stack", alpha=0.5
    )

    ax = g.ax_joint
    ax.tick_params(axis="both", labelsize=20)

    box_style_dict = dict(
        boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
    )

    def show_text(x, y, text):
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=box_style_dict,
        )

    show_text(0.95, 0.13, f"Train $R^2$ = {r2_score(y_train, train_pred):.3f}")
    show_text(0.95, 0.05, f"Test $R^2$ = {r2_score(y_test, test_pred):.3f}")
    show_text(0.90, 0.95, f"Model = {model_name}")
    ax.plot(
        [dataset["True"].min(), dataset["True"].max()],
        [dataset["True"].min(), dataset["True"].max()],
        c="black",
        alpha=0.5,
        linestyle="--",
        label="x=y",
    )
    ax.legend(fontsize=fontsize)
    ax.set_xlabel("True", fontsize=fontsize)
    ax.set_ylabel("Predicted", fontsize=fontsize)
    fig = ax.get_figure()
    ax2 = fig.add_axes([0.56, 0.23, 0.27, 0.27])
    shown_range = [0, 400]
    ax2.set_xlim(shown_range)
    ax2.set_ylim(shown_range)
    ax2.set_xticks(shown_range)
    ax2.set_yticks(shown_range)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(labelsize=10)
    sns.regplot(
        data=dataset_train,
        scatter=True,
        x="True",
        y="Predicted",
        ax=ax2,
        color=palette["Train"],
    )

    sns.regplot(
        data=dataset_test,
        scatter=True,
        x="True",
        y="Predicted",
        ax=ax2,
        color=palette["Test"],
    )
    ax2.plot(
        shown_range,
        shown_range,
        c="black",
        alpha=0.5,
        linestyle="--",
        label="x=y",
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    return g
