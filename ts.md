**(1) 求解以下的同余方程组：**

$$
\begin{cases}
x \equiv 1 \pmod{12} \\
x \equiv 4 \pmod{9}
\end{cases}
$$

**解答：**

首先，根据第一个同余方程 $$ x \equiv 1 \pmod{12} $$，设：

$$
x = 12k + 1，\quad k \in \mathbb{Z}
$$

将 $$ x $$ 代入第二个同余方程：

$$
12k + 1 \equiv 4 \pmod{9}
$$

因为 $$ 12 \equiv 3 \pmod{9} $$，所以：

$$
3k + 1 \equiv 4 \pmod{9}
$$

移项得：

$$
3k \equiv 3 \pmod{9}
$$

两边同时减去 1：

$$
3k \equiv 3 \pmod{9}
$$

两边同除以 3（注意到 $$ \gcd(3,9) = 3 $$）：

$$
k \equiv 1 \pmod{3}
$$

因此，$$ k = 3t + 1，\quad t \in \mathbb{Z} $$。

代回 $$ x $$ 的表达式：

$$
x = 12k + 1 = 12(3t + 1) + 1 = 36t + 13
$$

所以，$$ x \equiv 13 \pmod{36} $$。

**答：**$$ x \equiv 13 \pmod{36} $$。

---

**(2) 试说明以下的同余方程组无解：**

$$
\begin{cases}
x \equiv 1 \pmod{12} \\
x \equiv 2 \pmod{9}
\end{cases}
$$

**解答：**

同样，根据第一个同余方程 $$ x \equiv 1 \pmod{12} $$，设：

$$
x = 12k + 1，\quad k \in \mathbb{Z}
$$

将 $$ x $$ 代入第二个同余方程：

$$
12k + 1 \equiv 2 \pmod{9}
$$

由于 $$ 12 \equiv 3 \pmod{9} $$，所以：

$$
3k + 1 \equiv 2 \pmod{9}
$$

移项得：

$$
3k \equiv 1 \pmod{9}
$$

但是，$$ \gcd(3,9) = 3 $$，且 3 是左边 $$ 3k $$ 的因数，但不是右边 1 的因数。因此，同余方程 $$ 3k \equiv 1 \pmod{9} $$ 无解。

**答：**由于同余方程 $$ 3k \equiv 1 \pmod{9} $$ 无解，故原方程组无解。