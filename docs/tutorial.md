# Understanding a Minimal Language Model from Scratch: Hello World LLM Tutorial

> **Target audience**: Programmers who can write CRUD code (frontend/backend/fullstack) but have zero background in machine learning, neural networks, or LLMs.
>
> **What we'll do**: Walk through every line of a ~500-line character-level language model written in pure TypeScript. No ML frameworks — all the math is just `for` loops and basic arithmetic.

---

## Table of Contents

1. [What Does a Language Model Actually Do?](#1-what-does-a-language-model-actually-do)
2. [Character-Level Tokenizer: Turning Text into Numbers](#2-character-level-tokenizer-turning-text-into-numbers)
3. [Embeddings: Giving Each Character an "ID Card"](#3-embeddings-giving-each-character-an-id-card)
4. [Forward Pass: From Input to Prediction](#4-forward-pass-from-input-to-prediction)
   - 4.1 [Matrix Multiplication: Bundled Multiply-and-Add](#41-matrix-multiplication-bundled-multiply-and-add)
   - 4.2 [Activation Function tanh: Letting the Model Learn "Curves"](#42-activation-function-tanh-letting-the-model-learn-curves)
   - 4.3 [Softmax: Turning Scores into Probabilities](#43-softmax-turning-scores-into-probabilities)
5. [Loss Function: How Wrong Is the Model?](#5-loss-function-how-wrong-is-the-model)
6. [Backpropagation: Telling Each Parameter How to Adjust](#6-backpropagation-telling-each-parameter-how-to-adjust)
   - 6.1 [Derivatives/Gradients: The "Slope" of a Function](#61-derivativesgradients-the-slope-of-a-function)
   - 6.2 [The Chain Rule: One Link After Another](#62-the-chain-rule-one-link-after-another)
   - 6.3 [Backpropagation in the Code, Layer by Layer](#63-backpropagation-in-the-code-layer-by-layer)
7. [SGD Optimizer: Walking Downhill](#7-sgd-optimizer-walking-downhill)
8. [Training Loop: Practice Makes Perfect](#8-training-loop-practice-makes-perfect)
9. [Text Generation: Making the Model Talk](#9-text-generation-making-the-model-talk)
10. [From This Toy to Real LLMs: What's the Gap?](#10-from-this-toy-to-real-llms-whats-the-gap)

---

## 1. What Does a Language Model Actually Do?

Think of a language model as a **super-powered autocomplete**.

When your phone keyboard suggests "weather" after you type "How's the" — that's essentially a simple language model: **given previous text, predict the probability of the next character (or token)**.

```
Input:  "To be, or not to b"
Model output:
  'e' → 92%    ← most likely
  'a' →  3%
  'o' →  1%
  ... remaining characters share the other 4%
```

ChatGPT and Claude do fundamentally the same thing — except their "characters" are larger tokens (word pieces), their models are millions of times bigger, and they've been fine-tuned for dialogue. But the core logic is identical: **read the preceding text, guess what comes next**.

What our Hello World LLM does:
- Training data: two Shakespeare soliloquies (~1,900 characters)
- Learning objective: given the previous 16 characters, predict the 17th
- After training: give it a prompt and it can "continue" in a Shakespeare-like style (though with plenty of misspellings — the model is tiny)

---

## 2. Character-Level Tokenizer: Turning Text into Numbers

Computers do arithmetic on numbers, not on characters like 'a' or 'b'. So the first step is building a **mapping table between characters and numbers**.

```typescript
const chars = Array.from(new Set(TRAINING_TEXT)).sort();
// chars = ['\n', ' ', "'", ',', '.', ':', ';', '?', 'A', 'B', ... 'z']

const stoi: Record<string, number> = {}; // string to integer
const itos: Record<number, string> = {}; // integer to string
chars.forEach((ch, i) => { stoi[ch] = i; itos[i] = ch; });
```

This is no different from creating a `char_id → char` lookup table in a database:

| char_id | char |
|---------|------|
| 0       | `\n` |
| 1       | ` `  |
| 2       | `'`  |
| 3       | `,`  |
| ...     | ...  |
| 42      | `z`  |

`encode` converts a string into an array of numbers; `decode` does the reverse:

```typescript
encode("To be") → [20, 31, 1, 18, 21]
decode([20, 31, 1, 18, 21]) → "To be"
```

**Vocabulary size (VOCAB_SIZE)** = the number of unique characters in the training text — about 43 here.

> **How real LLMs differ**: GPT doesn't tokenize individual characters. It uses a BPE (Byte Pair Encoding) algorithm to merge common character combinations into tokens. For example, "the" might be one token and "tion" another. This gives vocabulary sizes of 32K–100K, where each token represents several characters on average — much more efficient.

---

## 3. Embeddings: Giving Each Character an "ID Card"

Now each character has an ID number (0, 1, 2, ...), but a single integer carries too little information. We want to represent each character as a **multi-dimensional vector**, giving the model more "dimensions" to learn relationships between characters.

### What's a vector?

If you've written CSS, `rgb(255, 128, 0)` is a 3-dimensional vector — 3 numbers describing a color.

Embeddings work the same way: we use `N_EMBD = 32` numbers to describe each character. Initially these 32 numbers are random, but during training they **adjust automatically** so that:
- Similar characters (e.g., vowels a, e, i, o, u) end up closer together in 32-dimensional space
- Different characters end up farther apart

```typescript
// Token Embedding table: a VOCAB_SIZE × N_EMBD matrix
// Each row is one character's 32-dimensional "ID card"
const tokEmb = randn(VOCAB_SIZE * N_EMBD, 0.02);

// Position Embedding table: BLOCK_SIZE × N_EMBD
// Each row represents "this is the Nth position in the context window"
const posEmb = randn(BLOCK_SIZE * N_EMBD, 0.02);
```

**Why do we also need Position Embeddings?**

Imagine reading "ab" vs. "ba" — same two characters, different order, completely different meaning. If we only used character embeddings, the model couldn't distinguish order. Position embeddings give each position a unique vector, added to the character embedding, so the model knows "'a' at position 1" and "'a' at position 5" are different.

In the code, it works like this:

```
Final representation of character c at position t = tokEmb[c] + posEmb[t]
```

Then the vectors from all 16 positions are concatenated into one 16 × 32 = 512-dimensional vector — this is the input to the next layer.

> **Analogy**: Imagine a student information system. `tokEmb` is like "personal attributes" (height, weight, age...), and `posEmb` is like "seat-number metadata". The same student sitting in different seats gets a different combined description: "personal attributes + seat attributes".

---

## 4. Forward Pass: From Input to Prediction

The forward pass is **the process of computing from input to output** — data flows left to right through each layer of the model. Think of it as a multi-step data processing pipeline:

```
[16 characters] → Embedding → Concat into 512-dim vector → Hidden layer (512→128) → tanh → Output layer (128→43) → Softmax → Probabilities
```

### 4.1 Matrix Multiplication: Bundled Multiply-and-Add

This is the most fundamental operation in neural networks. Don't be intimidated by the word "matrix" — it's just a way of doing multiplications and additions in bulk.

**A concrete example**: Suppose you're building a "dish recommendation system" with 3 features (spiciness, sweetness, saltiness) and you want to compute preference scores for 2 people:

```
Dish features:  [spiciness=0.8, sweetness=0.3, saltiness=0.5]

Person A's taste weights: spicy×0.9 + sweet×0.1 + salty×0.4
Person B's taste weights: spicy×0.2 + sweet×0.8 + salty×0.3
```

Calculation:
```
Person A's score = 0.8×0.9 + 0.3×0.1 + 0.5×0.4 = 0.72 + 0.03 + 0.20 = 0.95
Person B's score = 0.8×0.2 + 0.3×0.8 + 0.5×0.3 = 0.16 + 0.24 + 0.15 = 0.55
```

That's a matrix multiplication! "Input vector × weight matrix = output vector" — each output element is the sum of all input elements multiplied by their corresponding weights.

In our model:

```typescript
// Hidden layer: 512-dim input → 128-dim output
// W1 is a 512×128 weight matrix (512×128 = 65,536 numbers)
// b1 is a 128-dim bias vector
for (let h = 0; h < HIDDEN_DIM; h++) {
  let sum = b1[h];  // bias: a baseline value
  for (let i = 0; i < INPUT_DIM; i++) {
    sum += flat[b * INPUT_DIM + i] * W1[i * HIDDEN_DIM + h];
    //     ^^^^^^^^input dim i^^^^^   ^^^^^^^^weight^^^^^^^^
  }
  hidden[b * HIDDEN_DIM + h] = Math.tanh(sum);
}
```

**Bias (b1)** is just an added constant, analogous to the `b` in the linear equation `y = kx + b`. Without it, every line must pass through the origin, limiting expressiveness.

### 4.2 Activation Function tanh: Letting the Model Learn "Curves"

If we only had matrix multiplications (which are linear transformations), no matter how many layers we stack, the result is equivalent to a single layer — because **a composition of linear functions is still linear**.

Analogy: `y = 3 × (2 × x)` is the same as `y = 6x` — two layers, but no added power.

The `tanh` (hyperbolic tangent) function squashes any number into the range -1 to 1:

```
tanh(-100) ≈ -1
tanh(0)    = 0
tanh(100)  ≈ 1
tanh(0.5)  ≈ 0.46
```

Its shape is an S-curve. By adding this "bend", the model can learn non-linear patterns — for example, "after this character, certain characters suddenly become much more likely."

> **Analogy**: Imagine you're adjusting an audio equalizer. A linear model without tanh is like only being able to uniformly boost or cut all frequencies. With tanh, you can create non-linear curves like "boost bass, cut treble."

### 4.3 Softmax: Turning Scores into Probabilities

After the hidden and output layers, we get VOCAB_SIZE (43) raw scores called **logits**. For example:

```
logits = [2.1, -0.5, 0.3, 1.8, -1.2, ...]  (43 numbers)
```

But these scores can be positive or negative and don't sum to 1, so they can't be used directly as probabilities. The softmax function converts them into a **valid probability distribution** (all positive, summing to 1):

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

If you haven't seen $e$ before — it's a mathematical constant approximately equal to 2.718. $e^x$ means "2.718 to the power of x". It has a nice property: it's always positive, and the larger x is, the faster $e^x$ grows.

**Walking through with concrete numbers**:

```
logits = [2.0, 1.0, 0.1]

Step 1: Compute e to each power
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105

Step 2: Sum them
  total = 7.389 + 2.718 + 1.105 = 11.212

Step 3: Divide each by the total
  probabilities = [7.389/11.212, 2.718/11.212, 1.105/11.212]
                = [0.659, 0.242, 0.099]
                (sum = 1.0 ✓)
```

**Numerical stability trick**: The code first subtracts the maximum value `maxVal`, because $e^{1000}$ would overflow (become Infinity), but $e^0$ won't. Mathematically, `softmax(x - max)` produces the same result as `softmax(x)`.

```typescript
// Find the maximum value
let maxVal = -Infinity;
for (let v = 0; v < VOCAB_SIZE; v++)
  maxVal = Math.max(maxVal, logits[b * VOCAB_SIZE + v]);

// Subtract max before exp — prevents overflow
let sumExp = 0;
for (let v = 0; v < VOCAB_SIZE; v++) {
  probs[b * VOCAB_SIZE + v] = Math.exp(logits[b * VOCAB_SIZE + v] - maxVal);
  sumExp += probs[b * VOCAB_SIZE + v];
}
// Normalize
for (let v = 0; v < VOCAB_SIZE; v++)
  probs[b * VOCAB_SIZE + v] /= sumExp;
```

---

## 5. Loss Function: How Wrong Is the Model?

The model has produced a probability distribution over 43 characters, but there's only one correct next character. **Loss** measures how "wrong" the model's prediction is.

We use **Cross-Entropy Loss**, which is remarkably simple:

$$\text{Loss} = -\log(\text{probability the model assigned to the correct answer})$$

Examples:

| Scenario | Probability of correct answer | Loss |
|----------|------------------------------|------|
| Model is confident, gave 90% | -log(0.9) = 0.105 | Low (good) |
| Model is uncertain, gave 50% | -log(0.5) = 0.693 | Medium |
| Model guessed wrong, only 1% | -log(0.01) = 4.605 | High (bad) |

**Why use log?**

- If probability is close to 1 (model guessed right), -log(1) = 0, zero penalty
- If probability is close to 0 (model completely wrong), -log(0) → ∞, infinite penalty
- This penalizes much more aggressively than a simple `1 - p`, pushing the model to learn more effectively

```typescript
function computeLoss(cache: ForwardCache): number {
  let totalLoss = 0;
  for (let b = 0; b < cache.B; b++) {
    // Find the probability assigned to the correct answer, take -log
    totalLoss -= Math.log(cache.probs[b * VOCAB_SIZE + cache.targets[b]] + 1e-10);
    //                                                                    ^^^^^^
    //                                          add a tiny value to prevent log(0) = -Infinity
  }
  return totalLoss / cache.B;  // average over the batch
}
```

The goal of training is to make this loss smaller and smaller.

---

## 6. Backpropagation: Telling Each Parameter How to Adjust

This is the most critical — and most challenging — part of the whole model. But don't worry, we'll start from the basics.

### 6.1 Derivatives/Gradients: The "Slope" of a Function

Imagine you're standing on a mountain and want to reach the lowest point in the valley. You don't need to see a global map — you only need to feel **the slope under your feet**: whichever direction goes downhill, walk that way.

**A derivative is a slope**. For a function $f(x)$, the derivative $f'(x)$ tells you:

> If x increases by a tiny amount, how much does f(x) change?

Example: $f(x) = x^2$
- At $x = 3$, the derivative = $2 \times 3 = 6$, meaning for each unit increase in x, f(x) increases by about 6
- At $x = -2$, the derivative = $2 \times (-2) = -4$, meaning for each unit increase in x, f(x) actually decreases by 4

**Gradient** is just the multi-dimensional version of a derivative. If a function has many input parameters, the gradient is the derivative with respect to each parameter separately, forming a vector that tells you **which direction to adjust each parameter to decrease the function fastest**.

Our model has ~44,000 parameters (all the embeddings, W1, b1, W2, b2 combined). We need to know: **if we slightly tweak each parameter, how does the loss change?** That's what backpropagation computes.

### 6.2 The Chain Rule: One Link After Another

Our model is a multi-layer computation chain:

```
input → embedding → hidden layer → tanh → output layer → softmax → loss
```

The **chain rule** says: if $y = f(g(x))$, then the derivative of $y$ with respect to $x$ = derivative of $f$ with respect to $g$ × derivative of $g$ with respect to $x$.

A real-life example to build intuition:

> Exchange rates: 1 USD = 7 CNY, 1 USD = 0.9 EUR
>
> Question: 1 EUR = how many CNY?
>
> Answer: First "EUR → USD" multiply by 1/0.9, then "USD → CNY" multiply by 7 = 7/0.9 ≈ 7.78
>
> That's the chain rule: **multiply the rate of change at each step together**.

In backpropagation:
1. First compute the derivative of Loss with respect to softmax output (slope of the last step)
2. Multiply by the derivative of softmax with respect to logits (slope of the second-to-last step)
3. Multiply by the derivative of logits with respect to W2 (keep pushing backwards)
4. ...all the way back to the initial embeddings

That's why it's called "back" propagation — we work from the last layer to the first.

### 6.3 Backpropagation in the Code, Layer by Layer

#### Step 1: Gradient of Softmax + Cross-Entropy

Here's a famously elegant simplification — after combining softmax and cross-entropy, the gradient formula is beautifully simple:

$$\frac{\partial \text{Loss}}{\partial \text{logits}_i} = \text{probs}_i - \mathbb{1}[i = \text{target}]$$

In plain English: **gradient = model's predicted probability - correct answer**.

If the correct answer is character 5:
```
probs   = [0.1, 0.05, 0.02, 0.03, 0.1, 0.6, ...]  ← model's prediction
one_hot = [0,   0,    0,    0,    0,   1,   ...]  ← correct answer (only position 5 is 1)
gradient = [0.1, 0.05, 0.02, 0.03, 0.1, -0.4, ...]  ← the difference
```

Corresponding code:

```typescript
const dLogits = new Float64Array(B * VOCAB_SIZE);
for (let b = 0; b < B; b++) {
  for (let v = 0; v < VOCAB_SIZE; v++)
    dLogits[b * VOCAB_SIZE + v] = c.probs[b * VOCAB_SIZE + v] / B;
  dLogits[b * VOCAB_SIZE + c.targets[b]] -= 1.0 / B;
  // Dividing by B because loss is averaged over the batch
}
```

#### Step 2: Gradients for the Output Layer (W2, b2)

Mathematically, `logits = hidden × W2 + b2`, so:

$$\frac{\partial \text{Loss}}{\partial W2} = \text{hidden}^T \times d\text{Logits}$$

$$\frac{\partial \text{Loss}}{\partial b2} = \text{sum}(d\text{Logits}, \text{along batch dimension})$$

Intuition: the larger a hidden neuron's activation, the more its corresponding weight needs to adjust (because it has more influence).

```typescript
// dW2[h][v] = sum over all batches of hidden[h] × dLogits[v]
const dW2 = new Float64Array(HIDDEN_DIM * VOCAB_SIZE);
for (let h = 0; h < HIDDEN_DIM; h++) {
  for (let v = 0; v < VOCAB_SIZE; v++) {
    let s = 0;
    for (let b = 0; b < B; b++)
      s += c.hidden[b * HIDDEN_DIM + h] * dLogits[b * VOCAB_SIZE + v];
    dW2[h * VOCAB_SIZE + v] = s;
  }
}
```

#### Step 3: Propagating the Gradient Through tanh

The derivative of tanh has a convenient property:

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

This means if you already know the value of `tanh(x)` (which we computed and stored in `hidden` during the forward pass), you can get the derivative directly with `1 - hidden²` — no recomputation needed.

```typescript
const dHiddenPre = new Float64Array(B * HIDDEN_DIM);
for (let i = 0; i < B * HIDDEN_DIM; i++) {
  const t = c.hidden[i];         // already the post-tanh value
  dHiddenPre[i] = dHidden[i] * (1 - t * t);  // multiply by tanh's derivative
}
```

#### Step 4: Gradients for the Hidden Layer (W1, b1)

Almost the same structure as Step 2, just with different dimensions:

```typescript
// dW1 = flat^T × dHiddenPre
const dW1 = new Float64Array(INPUT_DIM * HIDDEN_DIM);
for (let i = 0; i < INPUT_DIM; i++) {
  for (let h = 0; h < HIDDEN_DIM; h++) {
    let s = 0;
    for (let b = 0; b < B; b++)
      s += c.flat[b * INPUT_DIM + i] * dHiddenPre[b * HIDDEN_DIM + h];
    dW1[i * HIDDEN_DIM + h] = s;
  }
}
```

#### Step 5: Gradients for Embeddings

Embeddings aren't a matrix multiplication — they're a table lookup. So gradient propagation isn't a matrix operation either; instead, we **accumulate the gradient back to the corresponding entries**:

```typescript
for (let b = 0; b < B; b++) {
  for (let t = 0; t < BLOCK_SIZE; t++) {
    const tokIdx = c.inputs[b][t];
    for (let d = 0; d < N_EMBD; d++) {
      const g = dFlat[b * INPUT_DIM + t * N_EMBD + d];
      dTokEmb[tokIdx * N_EMBD + d] += g;  // accumulate! same character appearing multiple times gets gradients summed
      dPosEmb[t * N_EMBD + d] += g;
    }
  }
}
```

---

## 7. SGD Optimizer: Walking Downhill

Now that we have the gradient (slope for each parameter), the next step is to **take a small step in the opposite direction of the slope** — because we want to decrease the loss, and the gradient points in the direction of increasing loss.

$$\text{parameter}_{\text{new}} = \text{parameter}_{\text{old}} - \text{learning rate} \times \text{gradient}$$

This is **SGD (Stochastic Gradient Descent)** — "stochastic" because each update uses only a small batch of data rather than the entire dataset to estimate the gradient.

```typescript
function sgdUpdate(g: Grads, lr: number): void {
  for (let i = 0; i < tokEmb.length; i++) tokEmb[i] -= lr * g.dTokEmb[i];
  for (let i = 0; i < W1.length; i++)     W1[i]     -= lr * g.dW1[i];
  // ... same for every parameter group
}
```

**Learning rate** is the step size. Too large and you'll "overshoot" the valley and oscillate back and forth; too small and you'll crawl along painfully slowly. The code uses a **Cosine Schedule**:

```
The learning rate follows a cosine curve over training:
- First 200 steps: linearly increase from 0 to the maximum of 0.05 (warmup / slow start)
- After that: cosine decay down to a minimum of 0.0005
```

Intuition: early on the parameters are still random, so big steps are fine; later the model is nearly converged and needs small, precise adjustments.

### Gradient Clipping

Sometimes gradients suddenly become very large ("gradient explosion"), and one oversized step can ruin the model. Gradient clipping sets an upper limit: if the total magnitude of the gradient exceeds `GRAD_CLIP = 5.0`, the entire gradient vector is proportionally scaled down to length 5.0.

```typescript
function clipGrads(g: Grads, maxNorm: number): void {
  // Compute the total gradient magnitude (L2 norm)
  let normSq = 0;
  for (const arr of arrays)
    for (let i = 0; i < arr.length; i++) normSq += arr[i] * arr[i];
  const norm = Math.sqrt(normSq);

  // If it exceeds the limit, scale down proportionally
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (const arr of arrays)
      for (let i = 0; i < arr.length; i++) arr[i] *= scale;
  }
}
```

---

## 8. Training Loop: Practice Makes Perfect

The entire training process is like a student doing practice problems over and over:

```
Repeat 5,000 times:
  1. Randomly sample a batch of training data (32 examples of "16 chars → 1 char")
  2. Forward pass: let the model guess → get a probability distribution
  3. Compute loss: measure how wrong the guess is
  4. Backpropagation: compute the gradient of every parameter (which way to adjust)
  5. SGD update: slightly adjust all parameters
```

```typescript
for (let step = 0; step < NUM_STEPS; step++) {
  const lr = ...;  // compute current learning rate
  const { inputs, targets } = sampleBatch();    // 1. sample problems
  const cache = forward(inputs, targets);        // 2. attempt answers
  const loss = computeLoss(cache);               // 3. grade the attempt
  const grads = backward(cache);                 // 4. analyze mistakes
  clipGrads(grads, GRAD_CLIP);                   // 4.5 prevent gradient explosion
  sgdUpdate(grads, lr);                          // 5. improve
}
```

As training progresses, the loss gradually decreases:
```
Step     0 │ Loss: 3.7612   ← random guessing at first, loss is high
Step  1000 │ Loss: 2.1543   ← learned some patterns
Step  3000 │ Loss: 1.5234   ← getting better
Step  4999 │ Loss: 1.2876   ← roughly converged
```

> **Theoretical minimum loss**: If the model perfectly learned the character distribution in the training data, the loss would be around 1.0–1.5 (because natural language is inherently uncertain — the same prefix can be followed by many reasonable characters).

---

## 9. Text Generation: Making the Model Talk

After training, we can use the model to generate text **autoregressively**. "Autoregressive" means: **the model generates one character, appends it to the input, predicts the next character, and repeats**.

```
Context window (16 chars)        → Model predicts → Sample → New character
["T","o"," ","b","e",...]       → probs          → ","   → append to context
["o"," ","b","e",",",...]       → probs          → " "   → append to context
[" ","b","e",","," ",...]       → probs          → "o"   → append to context
...
```

### Temperature Sampling

The model outputs a probability distribution, and we need to **randomly sample** a character from it. Temperature controls the randomness:

```
Raw logits = [2.0, 1.0, 0.1]

Temperature = 1.0 → softmax([2.0, 1.0, 0.1])   = [0.659, 0.242, 0.099]  (normal)
Temperature = 0.5 → softmax([4.0, 2.0, 0.2])   = [0.843, 0.114, 0.043]  (more focused, more conservative)
Temperature = 2.0 → softmax([1.0, 0.5, 0.05])  = [0.414, 0.251, 0.335]  (more uniform, more random)
```

How it works: divide the logits by the temperature before applying softmax:
- **Temperature < 1**: Effectively magnifies differences, making high-probability characters stand out more (generates "safer" but more boring text)
- **Temperature > 1**: Effectively shrinks differences, making all characters closer in probability (generates more "creative" but potentially garbled text)
- **Temperature → 0**: In the extreme, always picks the highest-probability character (greedy decoding)

The code uses Temperature = 0.8, slightly on the conservative side:

```typescript
// Divide by temperature
for (let v = 0; v < VOCAB_SIZE; v++)
  scaled[v] = cache.logits[v] / temperature;

// softmax
// ...

// Randomly sample from the probability distribution
const r = Math.random();  // random number in [0, 1)
let cum = 0;
let next = 0;
for (let v = 0; v < VOCAB_SIZE; v++) {
  cum += scaled[v];       // cumulative probability
  if (cum >= r) { next = v; break; }  // whichever bucket r falls into, pick that character
}
```

This sampling method is like a prize wheel — the higher a character's probability, the larger its slice of the wheel, and the more likely it is to be selected.

---

## 10. From This Toy to Real LLMs: What's the Gap?

Our model has ~44K parameters; GPT-4 is estimated to have over a trillion. But the core ideas are the same. The main differences:

| Aspect | Our Hello World | Real LLMs (e.g., GPT-4) |
|--------|----------------|--------------------------|
| **Core architecture** | Single hidden layer MLP | Transformer (multi-layer Self-Attention) |
| **Key innovation** | None | Self-Attention: each position can "see" all other positions |
| **Tokenizer** | Single character (~43 tokens) | BPE/SentencePiece (32K–100K tokens) |
| **Parameters** | ~44,000 | Billions to trillions |
| **Training data** | 1,900 characters of Shakespeare | Trillions of tokens from the internet |
| **Context window** | 16 characters | Thousands to hundreds of thousands of tokens |
| **Optimizer** | SGD | AdamW (with momentum and adaptive learning rates) |
| **Hardware** | CPU, takes seconds | Thousands of GPUs, weeks to months of training |
| **Activation function** | tanh | GeLU / SwiGLU |
| **Extra techniques** | None | Layer Norm, Dropout, KV Cache, RLHF/DPO... |

**The most critical difference is Self-Attention**:

Our model brute-force concatenates the embeddings of 16 characters into a single vector — each position's information is fixed. A Transformer's Self-Attention lets each position dynamically "attend to" other positions, deciding where to gather information from. This is the core contribution of the paper "Attention Is All You Need."

But even in a Transformer, the fundamental training process is identical to ours:

1. Take a piece of text, mask the last token
2. Forward pass — predict the masked token
3. Compute loss
4. Backpropagation — update parameters
5. Repeat until loss is low enough

Now that you understand this 500-line Hello World, you understand the skeleton of an LLM. The rest is just swapping each component for a more powerful version, stacking more layers, and feeding in more data.

---

## Appendix: Quick Reference for Key Math Notation

| Symbol | Meaning | Analogy |
|--------|---------|---------|
| $x$ | Input | A function's argument |
| $W$ | Weight matrix | "Strength coefficient" for each connection |
| $b$ | Bias | The `b` in `y = kx + b` |
| $\tanh(x)$ | Activation function | Squashes numbers into [-1, 1] |
| $e^x$ | Exponential function | 2.718 to the power of x |
| $\log(x)$ | Natural logarithm | Inverse of $e^x$; $\log(e^x) = x$ |
| $\frac{\partial L}{\partial W}$ | Partial derivative / gradient | The slope of Loss with respect to W |
| $\sum$ | Summation | A for-loop with accumulation |
| $\prod$ | Product | A for-loop with multiplication |
| $\mathbb{1}[\text{condition}]$ | Indicator function | Returns 1 if condition is true, 0 otherwise |
