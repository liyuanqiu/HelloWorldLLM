import { readFileSync } from 'fs';
import { resolve } from 'path';

/**
 * 🤖 Hello World LLM — A Character-Level Neural Language Model
 *
 * Built entirely from scratch in TypeScript. No ML frameworks — just math.
 *
 * Architecture:
 *   Character Embedding + Position Embedding → Hidden Layer (tanh) → Output (softmax)
 *
 * This model learns to predict the next character given a window of previous
 * characters, similar in principle to how GPT predicts the next token.
 *
 * Everything is implemented from scratch:
 *   - Forward pass (embedding lookup, matrix multiply, tanh, softmax)
 *   - Backward pass (manual backpropagation / gradient computation)
 *   - SGD optimizer (stochastic gradient descent)
 *   - Autoregressive text generation with temperature sampling
 *
 * 🎮 GPU Note (RTX 4090):
 *   At this scale (~44K params), CPU is fine. To leverage your RTX 4090,
 *   the matrix multiplications below are the operations to offload to GPU
 *   via WebGPU compute shaders, gpu.js, or CUDA bindings. Scaling to
 *   millions of parameters would make GPU acceleration essential.
 */

// ═══════════════════════════════════════════════════════════════════════
// Hyperparameters
// ═══════════════════════════════════════════════════════════════════════

const BLOCK_SIZE   = 16;     // Context window: how many previous chars the model sees
const N_EMBD       = 32;     // Embedding dimension per character
const HIDDEN_DIM   = 128;    // Number of neurons in the hidden layer
const BATCH_SIZE   = 32;     // Training examples per gradient update
const LR_MAX       = 0.05;   // Peak learning rate
const LR_MIN       = 0.0005; // Final learning rate
const NUM_STEPS    = 5000;   // Number of training iterations
const GRAD_CLIP    = 5.0;    // Max gradient norm (prevents exploding gradients)
const TEMPERATURE  = 0.8;    // Generation temperature (lower = more conservative)
const GEN_LENGTH   = 500;    // Number of characters to generate

// ═══════════════════════════════════════════════════════════════════════
// Training Data (西游记 — public domain)
// ═══════════════════════════════════════════════════════════════════════

const TRAINING_TEXT = readFileSync(
  resolve(__dirname, '..', 'data', 'xiyouji.txt'), 'utf-8'
);

// ═══════════════════════════════════════════════════════════════════════
// Character-Level Tokenizer
// ═══════════════════════════════════════════════════════════════════════

const chars = Array.from(new Set(TRAINING_TEXT)).sort();
const VOCAB_SIZE = chars.length;

// Character ↔ index mappings
const stoi: Record<string, number> = {};
const itos: Record<number, string> = {};
chars.forEach((ch, i) => { stoi[ch] = i; itos[i] = ch; });

const encode = (s: string): number[] => [...s].map(c => {
  if (!(c in stoi)) throw new Error(`Unknown character '${c}' — not in training vocabulary`);
  return stoi[c];
});
const decode = (ids: number[]): string => ids.map(i => itos[i]).join('');

// Encode the full training text into integer indices
const data = encode(TRAINING_TEXT);

// ═══════════════════════════════════════════════════════════════════════
// Model Parameters (randomly initialized)
// ═══════════════════════════════════════════════════════════════════════

/** Random normal initialization via Box-Muller transform (produces two samples per pair) */
function randn(size: number, std: number): Float64Array {
  const arr = new Float64Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    arr[i] = r * Math.cos(2 * Math.PI * u2) * std;
    if (i + 1 < size) arr[i + 1] = r * Math.sin(2 * Math.PI * u2) * std;
  }
  return arr;
}

const INPUT_DIM = BLOCK_SIZE * N_EMBD; // Flattened input size

// Token embeddings: each unique character → a learned vector
const tokEmb = randn(VOCAB_SIZE * N_EMBD, 0.02);

// Position embeddings: each position in the context window → a learned vector
const posEmb = randn(BLOCK_SIZE * N_EMBD, 0.02);

// Hidden layer: W1 (INPUT_DIM × HIDDEN_DIM), b1 (HIDDEN_DIM)
const W1 = randn(INPUT_DIM * HIDDEN_DIM, Math.sqrt(2.0 / INPUT_DIM));
const b1 = new Float64Array(HIDDEN_DIM);

// Output layer: W2 (HIDDEN_DIM × VOCAB_SIZE), b2 (VOCAB_SIZE)
const W2 = randn(HIDDEN_DIM * VOCAB_SIZE, Math.sqrt(2.0 / HIDDEN_DIM));
const b2 = new Float64Array(VOCAB_SIZE);

const totalParams = tokEmb.length + posEmb.length + W1.length + b1.length + W2.length + b2.length;

// ═══════════════════════════════════════════════════════════════════════
// Forward Pass
// ═══════════════════════════════════════════════════════════════════════

interface ForwardCache {
  B: number;               // batch size
  inputs: number[][];      // (B, BLOCK_SIZE) character indices
  targets: number[];       // (B,) target character indices
  flat: Float64Array;      // (B, INPUT_DIM) flattened embeddings
  hidden: Float64Array;    // (B, HIDDEN_DIM) post-activation values
  probs: Float64Array;     // (B, VOCAB_SIZE) softmax probabilities
  logits: Float64Array;    // (B, VOCAB_SIZE) raw output scores
}

function forward(inputs: number[][], targets: number[]): ForwardCache {
  const B = inputs.length;

  // Step 1: Embed characters + add position embeddings, then flatten
  //   For each position t in the context window, look up the character's
  //   embedding vector and add the position's embedding vector.
  //   Then concatenate all positions into one flat vector.
  const flat = new Float64Array(B * INPUT_DIM);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < BLOCK_SIZE; t++) {
      const tokIdx = inputs[b][t];
      for (let d = 0; d < N_EMBD; d++) {
        flat[b * INPUT_DIM + t * N_EMBD + d] =
          tokEmb[tokIdx * N_EMBD + d] + posEmb[t * N_EMBD + d];
      }
    }
  }

  // Step 2: Hidden layer — h = tanh(flat × W1 + b1)
  //   This is where the model learns non-linear patterns in the data.
  //   tanh squashes values to [-1, 1], introducing non-linearity.
  const hidden = new Float64Array(B * HIDDEN_DIM);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < HIDDEN_DIM; h++) {
      let sum = b1[h];
      for (let i = 0; i < INPUT_DIM; i++) {
        sum += flat[b * INPUT_DIM + i] * W1[i * HIDDEN_DIM + h];
      }
      hidden[b * HIDDEN_DIM + h] = Math.tanh(sum);
    }
  }

  // Step 3: Output layer — logits = hidden × W2 + b2
  //   Projects the hidden state to a score for each character in the vocabulary.
  const logits = new Float64Array(B * VOCAB_SIZE);
  for (let b = 0; b < B; b++) {
    for (let v = 0; v < VOCAB_SIZE; v++) {
      let sum = b2[v];
      for (let h = 0; h < HIDDEN_DIM; h++) {
        sum += hidden[b * HIDDEN_DIM + h] * W2[h * VOCAB_SIZE + v];
      }
      logits[b * VOCAB_SIZE + v] = sum;
    }
  }

  // Step 4: Softmax → probabilities
  //   Softmax converts logits to a probability distribution.
  const probs = new Float64Array(B * VOCAB_SIZE);
  for (let b = 0; b < B; b++) {
    let maxVal = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++)
      maxVal = Math.max(maxVal, logits[b * VOCAB_SIZE + v]);

    let sumExp = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      probs[b * VOCAB_SIZE + v] = Math.exp(logits[b * VOCAB_SIZE + v] - maxVal);
      sumExp += probs[b * VOCAB_SIZE + v];
    }
    for (let v = 0; v < VOCAB_SIZE; v++)
      probs[b * VOCAB_SIZE + v] /= sumExp;
  }

  return { B, inputs, targets, flat, hidden, logits, probs };
}

/** Compute cross-entropy loss from a forward cache (separated so generation can skip it) */
function computeLoss(cache: ForwardCache): number {
  let totalLoss = 0;
  for (let b = 0; b < cache.B; b++) {
    totalLoss -= Math.log(cache.probs[b * VOCAB_SIZE + cache.targets[b]] + 1e-10);
  }
  return totalLoss / cache.B;
}

// ═══════════════════════════════════════════════════════════════════════
// Backward Pass — Manual Backpropagation
//
// Computes ∂Loss/∂param for every parameter by applying the chain rule
// in reverse through each layer. This is exactly what autograd frameworks
// (PyTorch, TensorFlow) do automatically.
// ═══════════════════════════════════════════════════════════════════════

interface Grads {
  dTokEmb: Float64Array;
  dPosEmb: Float64Array;
  dW1: Float64Array;
  db1: Float64Array;
  dW2: Float64Array;
  db2: Float64Array;
}

function backward(c: ForwardCache): Grads {
  const { B } = c;

  // ── Gradient of softmax + cross-entropy ──
  // The beautiful result: dL/dlogits = probs - one_hot(target)
  // This elegant formula is why cross-entropy + softmax is so popular.
  const dLogits = new Float64Array(B * VOCAB_SIZE);
  for (let b = 0; b < B; b++) {
    for (let v = 0; v < VOCAB_SIZE; v++)
      dLogits[b * VOCAB_SIZE + v] = c.probs[b * VOCAB_SIZE + v] / B;
    dLogits[b * VOCAB_SIZE + c.targets[b]] -= 1.0 / B;
  }

  // ── Output layer gradients ──
  // dW2 = hidden^T × dLogits
  const dW2 = new Float64Array(HIDDEN_DIM * VOCAB_SIZE);
  for (let h = 0; h < HIDDEN_DIM; h++) {
    for (let v = 0; v < VOCAB_SIZE; v++) {
      let s = 0;
      for (let b = 0; b < B; b++)
        s += c.hidden[b * HIDDEN_DIM + h] * dLogits[b * VOCAB_SIZE + v];
      dW2[h * VOCAB_SIZE + v] = s;
    }
  }

  const db2 = new Float64Array(VOCAB_SIZE);
  for (let v = 0; v < VOCAB_SIZE; v++) {
    let s = 0;
    for (let b = 0; b < B; b++) s += dLogits[b * VOCAB_SIZE + v];
    db2[v] = s;
  }

  // ── Propagate gradient through output layer ──
  // dHidden = dLogits × W2^T
  const dHidden = new Float64Array(B * HIDDEN_DIM);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < HIDDEN_DIM; h++) {
      let s = 0;
      for (let v = 0; v < VOCAB_SIZE; v++)
        s += dLogits[b * VOCAB_SIZE + v] * W2[h * VOCAB_SIZE + v];
      dHidden[b * HIDDEN_DIM + h] = s;
    }
  }

  // ── tanh derivative: d/dx tanh(x) = 1 - tanh²(x) ──
  const dHiddenPre = new Float64Array(B * HIDDEN_DIM);
  for (let i = 0; i < B * HIDDEN_DIM; i++) {
    const t = c.hidden[i];
    dHiddenPre[i] = dHidden[i] * (1 - t * t);
  }

  // ── Hidden layer gradients ──
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

  const db1 = new Float64Array(HIDDEN_DIM);
  for (let h = 0; h < HIDDEN_DIM; h++) {
    let s = 0;
    for (let b = 0; b < B; b++) s += dHiddenPre[b * HIDDEN_DIM + h];
    db1[h] = s;
  }

  // ── Propagate gradient through hidden layer ──
  // dFlat = dHiddenPre × W1^T
  const dFlat = new Float64Array(B * INPUT_DIM);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < INPUT_DIM; i++) {
      let s = 0;
      for (let h = 0; h < HIDDEN_DIM; h++)
        s += dHiddenPre[b * HIDDEN_DIM + h] * W1[i * HIDDEN_DIM + h];
      dFlat[b * INPUT_DIM + i] = s;
    }
  }

  // ── Embedding gradients ──
  // Scatter the flat gradient back into the embedding tables.
  // Each character's embedding gets the accumulated gradient from
  // all positions where that character appeared.
  const dTokEmb = new Float64Array(VOCAB_SIZE * N_EMBD);
  const dPosEmb = new Float64Array(BLOCK_SIZE * N_EMBD);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < BLOCK_SIZE; t++) {
      const tokIdx = c.inputs[b][t];
      for (let d = 0; d < N_EMBD; d++) {
        const g = dFlat[b * INPUT_DIM + t * N_EMBD + d];
        dTokEmb[tokIdx * N_EMBD + d] += g;
        dPosEmb[t * N_EMBD + d] += g;
      }
    }
  }

  return { dTokEmb, dPosEmb, dW1, db1, dW2, db2 };
}

// ═══════════════════════════════════════════════════════════════════════
// Gradient Clipping — prevents exploding gradients
// ═══════════════════════════════════════════════════════════════════════

function clipGrads(g: Grads, maxNorm: number): void {
  const arrays = [g.dTokEmb, g.dPosEmb, g.dW1, g.db1, g.dW2, g.db2];
  let normSq = 0;
  for (const arr of arrays)
    for (let i = 0; i < arr.length; i++) normSq += arr[i] * arr[i];

  const norm = Math.sqrt(normSq);
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (const arr of arrays)
      for (let i = 0; i < arr.length; i++) arr[i] *= scale;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// SGD Update — the simplest optimizer
// ═══════════════════════════════════════════════════════════════════════

function sgdUpdate(g: Grads, lr: number): void {
  for (let i = 0; i < tokEmb.length; i++) tokEmb[i] -= lr * g.dTokEmb[i];
  for (let i = 0; i < posEmb.length; i++) posEmb[i] -= lr * g.dPosEmb[i];
  for (let i = 0; i < W1.length; i++)     W1[i]     -= lr * g.dW1[i];
  for (let i = 0; i < b1.length; i++)     b1[i]     -= lr * g.db1[i];
  for (let i = 0; i < W2.length; i++)     W2[i]     -= lr * g.dW2[i];
  for (let i = 0; i < b2.length; i++)     b2[i]     -= lr * g.db2[i];
}

// ═══════════════════════════════════════════════════════════════════════
// Batch Sampling — randomly pick training examples
// ═══════════════════════════════════════════════════════════════════════

function sampleBatch(): { inputs: number[][]; targets: number[] } {
  const inputs: number[][] = [];
  const targets: number[] = [];
  for (let i = 0; i < BATCH_SIZE; i++) {
    const pos = Math.floor(Math.random() * (data.length - BLOCK_SIZE - 1));
    inputs.push(data.slice(pos, pos + BLOCK_SIZE));
    targets.push(data[pos + BLOCK_SIZE]);
  }
  return { inputs, targets };
}

// ═══════════════════════════════════════════════════════════════════════
// Text Generation — autoregressive character-by-character
// ═══════════════════════════════════════════════════════════════════════

function generate(prompt: string, length: number, temperature: number): string {
  let context = encode(prompt);

  // Pad with space character if prompt is shorter than context window
  const padIdx = stoi[' '] ?? 0;
  while (context.length < BLOCK_SIZE)
    context = [padIdx, ...context];
  if (context.length > BLOCK_SIZE)
    context = context.slice(-BLOCK_SIZE);

  let result = prompt;

  for (let i = 0; i < length; i++) {
    // Forward pass to get logits (no loss computed — targets are irrelevant)
    const cache = forward([context], [0]);

    // Apply temperature scaling to logits, then softmax
    const scaled = new Float64Array(VOCAB_SIZE);
    let maxL = -Infinity;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      scaled[v] = cache.logits[v] / temperature;
      maxL = Math.max(maxL, scaled[v]);
    }
    let sumExp = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      scaled[v] = Math.exp(scaled[v] - maxL);
      sumExp += scaled[v];
    }
    for (let v = 0; v < VOCAB_SIZE; v++) scaled[v] /= sumExp;

    // Sample from the probability distribution
    const r = Math.random();
    let cum = 0;
    let next = 0;
    for (let v = 0; v < VOCAB_SIZE; v++) {
      cum += scaled[v];
      if (cum >= r) { next = v; break; }
    }

    result += itos[next];
    // Slide the context window forward
    context = [...context.slice(1), next];
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════
// 🚀 Main — Train the model and generate text
// ═══════════════════════════════════════════════════════════════════════

console.log('');
console.log('🤖 Hello World LLM — Character-Level Neural Language Model');
console.log('═══════════════════════════════════════════════════════════');
console.log(`  Vocabulary : ${VOCAB_SIZE} unique characters`);
console.log(`  Context    : ${BLOCK_SIZE} characters`);
console.log(`  Embedding  : ${N_EMBD}-dimensional vectors`);
console.log(`  Hidden     : ${HIDDEN_DIM} neurons (tanh activation)`);
console.log(`  Parameters : ${totalParams.toLocaleString()}`);
console.log(`  Data       : ${data.length.toLocaleString()} characters of 西游记`);
console.log('');

// ── Training loop ──
console.log('📚 Training...');
const t0 = Date.now();

for (let step = 0; step < NUM_STEPS; step++) {
  // Cosine learning rate schedule: warm up then decay
  const warmup = 200;
  let lr: number;
  if (step < warmup) {
    lr = LR_MAX * (step / warmup);
  } else {
    const decay = (step - warmup) / (NUM_STEPS - warmup);
    lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + Math.cos(Math.PI * decay));
  }

  const { inputs, targets } = sampleBatch();
  const cache = forward(inputs, targets);
  const loss = computeLoss(cache);
  const grads = backward(cache);
  clipGrads(grads, GRAD_CLIP);
  sgdUpdate(grads, lr);

  if (step % 500 === 0 || step === NUM_STEPS - 1) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`  Step ${String(step).padStart(5)} │ Loss: ${loss.toFixed(4)} │ LR: ${lr.toFixed(4)} │ ${elapsed}s`);
  }
}

const trainTime = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`\n✅ Training complete in ${trainTime}s`);

// ── Generate text ──
console.log('\n✨ Generated text (the model continues from "悟空"):');
console.log('───────────────────────────────────────────────────────────');
console.log(generate('悟空', GEN_LENGTH, TEMPERATURE));
console.log('───────────────────────────────────────────────────────────');

// ── Show what the model learned ──
console.log('\n📊 What the model learned:');
console.log('   The model assigns probabilities to every possible next character.');
console.log('   After training, it has learned patterns like:');
console.log('   - Common character sequences (的, 了, 不, 一, ...)');
console.log('   - Punctuation patterns (，。！？)');
console.log('   - Line and paragraph structure');
console.log('');
console.log('🔬 To scale this up to a real LLM, you would:');
console.log('   1. Add self-attention (the key innovation of Transformers)');
console.log('   2. Stack multiple layers (GPT-3 has 96 layers)');
console.log('   3. Use sub-word tokenization (BPE) instead of characters');
console.log('   4. Train on billions of tokens with GPU acceleration');
console.log(`   5. Offload matrix multiplications to your RTX 4090 via WebGPU/CUDA`);
console.log('');
