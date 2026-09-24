# Resolution（归结）——用符号推演证明蕴含

## 1. 归结是什么？

**Resolution（归结）**是一种具体的 **Inference（形式推演）方法**。

它的核心思想是：

> **为了证明 $KB \models \alpha$，先假设结论 $\alpha$ 是错误的，即把 $\neg\alpha$ 加入 KB，然后不断进行符号推演。如果最终推出矛盾（空子句 $\Box$），说明这个假设不可能成立，因此原结论 $\alpha$ 必然成立。**

可以简单记成：

$$
\boxed{\text{归结 = 假设结论为假，然后通过符号运算寻找矛盾}}
$$

---

# 2. 为什么归结可以证明 Entailment？

前面已经知道 Entailment 的一个重要等价条件：

$$
\boxed{
KB\models\alpha
\iff
KB\land\neg\alpha\text{ is Unsatisfiable}
}
$$

也就是说，要证明：

$$
KB\models\alpha
$$

我们不一定直接证明 $\alpha$。

可以反过来：

> **把 $\neg\alpha$ 加入 KB，然后证明它们不可能同时成立。**

即证明：

$$
KB\land\neg\alpha
$$

是 **Unsatisfiable（不可满足）**。

而 **Resolution（归结）就是用来证明这种“不可满足”的一种具体推理方法。**

---

# 3. 最基本的归结规则

假设有两个子句：

$$
P\lor Q
$$

和：

$$
\neg P\lor R
$$

其中：

$$
P
$$

和：

$$
\neg P
$$

是一对**互补文字（Complementary Literals）**。

进行归结时，可以将这一对互补文字消去：

$$
\frac{
P\lor Q,\qquad
\neg P\lor R
}{
Q\lor R
}
$$

得到：

$$
\boxed{Q\lor R}
$$

这一步就叫一次 **Resolution（归结）**。

---

# 4. 为什么可以把 P 和 ¬P 消掉？

考虑：

$$
P\lor Q
$$

和：

$$
\neg P\lor R
$$

如果两个句子都必须为真，那么讨论 P：

### 情况一：P 为真

因为：

$$
\neg P=F
$$

为了让：

$$
\neg P\lor R
$$

成立，就必须：

$$
R=T
$$

所以：

$$
Q\lor R=T
$$

---

### 情况二：P 为假

为了让：

$$
P\lor Q
$$

成立，就必须：

$$
Q=T
$$

所以：

$$
Q\lor R=T
$$

---

因此无论 P 是真还是假：

$$
Q\lor R
$$

都必须成立。

所以：

$$
P\lor Q
$$

和：

$$
\neg P\lor R
$$

可以归结得到：

$$
\boxed{Q\lor R}
$$

---

# 5. 用“下雨 → 地面湿”的例子理解归结

定义：

- $P$：下雨了
- $Q$：地面湿了

知识库：

$$
KB=\{P,\;P\rightarrow Q\}
$$

现在希望证明：

$$
KB\models Q
$$

也就是：

> 已知“下雨了”和“如果下雨，那么地面会湿”，能不能确定“地面湿了”？

---

## 第一步：把 Entailment 转换成不可满足问题

根据：

$$
KB\models Q
\iff
KB\land\neg Q\text{ is Unsatisfiable}
$$

我们假设结论 Q 是假的：

$$
\neg Q
$$

于是现在检查：

$$
KB\land\neg Q
$$

是否可能成立。

---

# 6. 第二步：把 Implication 转换掉

知识库中有：

$$
P\rightarrow Q
$$

利用逻辑等价：

$$
P\rightarrow Q
\equiv
\neg P\lor Q
$$

因此原来的：

$$
KB=\{P,\;P\rightarrow Q\}
$$

可以写成：

$$
\{P,\;\neg P\lor Q\}
$$

再加入假设：

$$
\neg Q
$$

最终得到三个子句：

$$
①\quad P
$$

$$
②\quad \neg P\lor Q
$$

$$
③\quad \neg Q
$$

---

# 7. 第三步：开始 Resolution

现在有：

$$
P
$$

和：

$$
\neg P\lor Q
$$

其中存在一对互补文字：

$$
P
$$

和：

$$
\neg P
$$

将它们消掉：

$$
\frac{
P,\qquad
\neg P\lor Q
}{
Q
}
$$

得到：

$$
\boxed{Q}
$$

---

# 8. 第四步：继续归结

现在我们已经推出：

$$
Q
$$

但之前为了进行反证，我们又加入了：

$$
\neg Q
$$

所以：

$$
Q
$$

和：

$$
\neg Q
$$

再次构成一对互补文字。

进行归结：

$$
\frac{
Q,\qquad
\neg Q
}{
\Box
}
$$

得到：

$$
\boxed{\Box}
$$

---

# 9. 空子句 □ 是什么意思？

$\Box$ 称为：

**Empty Clause（空子句）**

它表示：

> **矛盾 / False / 不可能满足。**

因为我们最终推出：

$$
\Box
$$

说明：

$$
KB\land\neg Q
$$

不存在任何 Model 可以满足。

因此：

$$
KB\land\neg Q
$$

是：

$$
\boxed{\text{Unsatisfiable}}
$$

根据：

$$
KB\models Q
\iff
KB\land\neg Q\text{ is Unsatisfiable}
$$

最终得到：

$$
\boxed{KB\models Q}
$$

证明完成。

---

# 10. 整个归结过程

整个过程可以串成：

    想证明：

    KB ⊨ α

        ↓

    不直接证明 α

        ↓

    假设 α 不成立

        ↓

    将 ¬α 加入 KB

        ↓

    KB ∧ ¬α

        ↓

    转换成适合归结的子句形式

        ↓

    寻找：

    P 和 ¬P

    这样的互补文字

        ↓

    不断进行 Resolution

        ↓

    如果最终得到：

    □

        ↓

    说明产生矛盾

        ↓

    KB ∧ ¬α
    Unsatisfiable

        ↓

    因此：

    KB ⊨ α

---

# 11. 归结本质上就是反证法

归结法的整体思路和数学中的**反证法**非常相似。

数学中的反证法：

> 想证明 A，就先假设 $\neg A$，如果最终推出矛盾，就说明 A 必须成立。

归结法：

> 想证明 $KB\models\alpha$，就先把 $\neg\alpha$ 加入 KB，如果通过归结最终推出空子句 $\Box$，就说明 $\alpha$ 必须成立。

因此可以记成：

$$
\boxed{
\text{Resolution}
=
\text{反证思想}
+
\text{形式化符号推演}
}
$$

---

# 12. Model Checking 和 Resolution 的区别

前面学习 Entailment 时，可以使用 **Model Checking / 枚举法**。

现在又出现了 **Resolution / 归结法**。

两者解决的目标类似，但方法完全不同。

| 方法 | 核心思想 |
|---|---|
| **Model Checking / 枚举** | 把所有可能世界列出来逐个检查 |
| **Resolution / 归结** | 假设结论为假，通过符号运算寻找矛盾 |
| **Entailment** | 要判断的逻辑关系 |
| **Inference** | 实际执行推理的算法 |
| **Resolution** | Inference 的一种具体实现方法 |

---

# 13. 用之前的知识体系理解归结

之前学习的逻辑体系是：

    Syntax
       ↓
    Semantics
       ↓
    Model
       ↓
    KB
       ↓
    Entailment
       ↓
    Inference

其中：

### Entailment

解决：

> **什么样的结论才算逻辑上必然成立？**

例如：

$$
KB\models\alpha
$$

---

### Inference

解决：

> **计算机具体怎么把这个结论算出来？**

而：

**Resolution 就是一种具体的 Inference 方法。**

所以：

$$
\boxed{
Resolution \subseteq Inference
}
$$

可以理解成：

> **Entailment 定义“什么叫推对了”；Resolution 提供一种“怎么把它推出来”的方法。**

---

# 14. 归结与之前两个等价条件的关系

之前学习：

$$
\boxed{
KB\models\alpha
}
$$

等价于：

$$
\boxed{
KB\rightarrow\alpha
\text{ is Valid}
}
$$

又等价于：

$$
\boxed{
KB\land\neg\alpha
\text{ is Unsatisfiable}
}
$$

归结法主要利用的是第三种形式：

$$
\boxed{
KB\models\alpha
\iff
KB\land\neg\alpha
\text{ is Unsatisfiable}
}
$$

因此：

    KB ⊨ α
        ↓
    KB ∧ ¬α
        ↓
    Resolution
        ↓
    □
        ↓
    Unsatisfiable
        ↓
    KB ⊨ α 成立

---

# 15. 最终一句话理解

> **归结（Resolution）是一种具体的形式推演算法：为了证明 $KB\models\alpha$，先把 $\neg\alpha$ 加入 KB，再不断消去互补文字；如果最终推出空子句 $\Box$，说明 $KB\land\neg\alpha$ 不可满足，因此原来的 $\alpha$ 必然被 KB 蕴含。**

最简记忆：

$$
\boxed{
KB\models\alpha
\Rightarrow
加入\ \neg\alpha
\Rightarrow
不断归结
\Rightarrow
\Box
\Rightarrow
证明成功
}
$$

或者直接记：

> **归结 = 假设结论为假 → 不断消去互补文字 → 推出空子句 → 产生矛盾 → 原结论成立。**