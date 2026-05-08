import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Backbone = "dense" | "resnet";

// ─── Pipeline stages ─────────────────────────────────────────────────────────

const STAGES = (backbone: Backbone) => {
  const dim = backbone === "dense" ? 1024 : 2048;
  return [
    {
      id: "input",
      label: "Input",
      shape: [1, 3, 224, 224],
      note: "RGB · ImageNet-norm",
      color: "var(--foreground)",
      heatmapSeed: 0.1,
    },
    {
      id: "backbone",
      label: backbone === "dense" ? "DenseNet121" : "ResNet50",
      shape: [1, dim, 7, 7],
      note: `3ch→${dim}ch, 224→7 px`,
      color: "var(--pos)",
      heatmapSeed: 0.35,
    },
    {
      id: "pos",
      label: "Pos Attention",
      shape: [1, dim, 7, 7],
      note: "γ·Softmax(QKᵀ/√d)V + x",
      color: "var(--pos)",
      heatmapSeed: 0.55,
    },
    {
      id: "chan",
      label: "Chan Attention",
      shape: [1, dim, 7, 7],
      note: "γ·Softmax(CCᵀ)·x + x",
      color: "var(--chan)",
      heatmapSeed: 0.68,
    },
    {
      id: "type",
      label: "Type Attention",
      shape: [1, dim, 7, 7],
      note: "Σ SE(Conv_k)·σ(gate) + x",
      color: "var(--type)",
      heatmapSeed: 0.78,
    },
    {
      id: "fusion",
      label: "Fusion",
      shape: [1, dim, 7, 7],
      note: "(pos + chan + type) / 3",
      color: "var(--fusion)",
      heatmapSeed: 0.85,
    },
    {
      id: "gap",
      label: "GAP → Flatten",
      shape: [1, dim],
      note: `${dim}×7×7 → ${dim}`,
      color: "var(--foreground)",
      heatmapSeed: 0.92,
    },
    {
      id: "head_type",
      label: "Head 1 · Type",
      shape: [1, 7],
      note: "512→256→7 · softmax",
      color: "var(--head-type)",
      heatmapSeed: 0.97,
    },
    {
      id: "head_shape",
      label: "Head 2 · Shape",
      shape: [1, 4],
      note: "256→4 · softmax",
      color: "var(--head-shape)",
      heatmapSeed: 0.97,
    },
    {
      id: "head_color",
      label: "Head 3 · Color",
      shape: [1, 2],
      note: "128→2 · softmax",
      color: "var(--head-color)",
      heatmapSeed: 0.97,
    },
  ];
};

type Stage = ReturnType<typeof STAGES>[number];

// ─── Helpers ─────────────────────────────────────────────────────────────────

const HEAD_CLASSES: Record<string, string[]> = {
  head_type: ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "Type 7"],
  head_shape: ["Constipation", "Normal", "Mild Diarrhea", "Liquid"],
  head_color: ["Brown", "Yellow"],
};

function shapeChanged(a: number[], b: number[]) {
  if (a.length !== b.length) return true;
  return a.some((v, i) => v !== b[i]);
}

function fmtBytes(shape: number[]) {
  const n = shape.reduce((a, b) => a * b, 1) * 4;
  return n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${(n / 1024).toFixed(1)} KB`;
}

// ─── Heatmap ─────────────────────────────────────────────────────────────────

function drawHeatmap(canvas: HTMLCanvasElement, seed: number, color: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width: W, height: H } = canvas;
  const tmp = document.createElement("div");
  tmp.style.cssText = `color:${color};display:none`;
  document.body.appendChild(tmp);
  const rgb = getComputedStyle(tmp).color.match(/\d+/g) ?? ["150", "100", "200"];
  document.body.removeChild(tmp);
  const [r, g, b] = rgb.map(Number);
  let s = seed * 999983;
  const rand = () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
  const data = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const x = i % W,
      y = Math.floor(i / W);
    const dist = Math.sqrt((x - W / 2) ** 2 + (y - H / 2) ** 2) / (W * 0.7);
    const v = Math.min(1, Math.max(0, 1 - dist) * seed + rand() * 0.3) * 255;
    data.data[i * 4] = (r / 255) * v;
    data.data[i * 4 + 1] = (g / 255) * v;
    data.data[i * 4 + 2] = (b / 255) * v;
    data.data[i * 4 + 3] = v * 0.85 + 40;
  }
  ctx.putImageData(data, 0, 0);
}

function HeatmapCanvas({
  seed,
  color,
  size = 56,
  active,
}: {
  seed: number;
  color: string;
  size?: number;
  active: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!active || !ref.current) return;
    const t = setTimeout(() => drawHeatmap(ref.current!, seed, color), 80);
    return () => clearTimeout(t);
  }, [active, seed, color]);
  return (
    <canvas
      ref={ref}
      width={size}
      height={size}
      className="rounded-lg"
      style={{
        imageRendering: "pixelated",
        border: `1.5px solid ${color}50`,
        opacity: active ? 1 : 0.18,
        transition: "opacity 0.4s",
      }}
    />
  );
}

// ─── Architecture overview strip ─────────────────────────────────────────────

const ARCH_GROUPS = [
  { label: "Input", ids: ["input"] },
  { label: "Backbone", ids: ["backbone"] },
  { label: "Attention ×3", ids: ["pos", "chan", "type"] },
  { label: "Fusion", ids: ["fusion"] },
  { label: "GAP", ids: ["gap"] },
  { label: "Heads ×3", ids: ["head_type", "head_shape", "head_color"] },
];

function ArchOverview({ activeId, stages }: { activeId: string; stages: Stage[] }) {
  const colorFor = (ids: string[]) =>
    stages.find((s) => ids.includes(s.id))?.color ?? "var(--foreground)";

  return (
    <div className="mb-5 flex items-center gap-1 overflow-x-auto pb-1">
      {ARCH_GROUPS.map((g, gi) => {
        const color = colorFor(g.ids);
        const isActive = g.ids.includes(activeId);
        return (
          <div key={g.label} className="flex shrink-0 items-center gap-1">
            {gi > 0 && (
              <svg width="16" height="12" viewBox="0 0 16 12" fill="none">
                <path
                  d="M1 6 H11 M7 2 L11 6 L7 10"
                  stroke={isActive ? color : "var(--border)"}
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            )}
            <motion.div
              animate={{ scale: isActive ? 1.04 : 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 28 }}
              className="rounded-lg border px-2.5 py-1.5"
              style={{
                borderColor: isActive ? color : "var(--border)",
                background: isActive ? `${color}14` : "var(--surface-soft)",
              }}
            >
              <span
                className="font-display block text-[10px] font-bold"
                style={{ color: isActive ? color : "var(--ink-mute)" }}
              >
                {g.label}
              </span>
            </motion.div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Pipeline rail (left column) ─────────────────────────────────────────────

function PipelineRail({
  stages,
  activeIndex,
  onSelect,
}: {
  stages: Stage[];
  activeIndex: number;
  onSelect: (i: number) => void;
}) {
  return (
    <div className="flex flex-col">
      {stages.map((stage, i) => {
        const isActive = i === activeIndex;
        const isDone = i < activeIndex;
        const changed = i > 0 && shapeChanged(stages[i - 1].shape, stage.shape);

        return (
          <div key={stage.id} className="relative flex items-start gap-3">
            {/* Vertical connector line */}
            {i < stages.length - 1 && (
              <div
                className="absolute left-[13px] z-0"
                style={{
                  top: 28,
                  bottom: -4,
                  width: 2,
                  borderRadius: 2,
                  background: isDone ? `${stage.color}55` : "var(--border)",
                  transition: "background 0.3s",
                }}
              />
            )}

            {/* Step circle */}
            <motion.button
              onClick={() => onSelect(i)}
              whileHover={{ scale: 1.12 }}
              whileTap={{ scale: 0.92 }}
              className="relative z-10 mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[10px] font-bold transition-shadow focus:outline-none"
              style={{
                background: isActive
                  ? stage.color
                  : isDone
                    ? "var(--surface-soft)"
                    : "var(--surface-soft)",
                color: isActive ? "#fff" : isDone ? stage.color : "var(--ink-mute)",
                border: `2px solid ${isActive ? stage.color : isDone ? `${stage.color}70` : "var(--border)"}`,
                boxShadow: isActive ? `0 0 0 4px ${stage.color}22` : "none",
                transition: "all 0.25s",
              }}
            >
              {isDone ? "✓" : i + 1}
            </motion.button>

            {/* Row label */}
            <button
              onClick={() => onSelect(i)}
              className="mb-4 flex flex-1 flex-col gap-0.5 text-left transition-opacity hover:opacity-70 focus:outline-none"
            >
              <span
                className="text-[12px] font-semibold leading-snug"
                style={{
                  color: isActive ? stage.color : isDone ? "var(--foreground)" : "var(--ink-mute)",
                }}
              >
                {stage.label}
              </span>
              <div className="flex flex-wrap items-center gap-1.5">
                <code
                  className="rounded-md px-1.5 py-0.5 text-[9px]"
                  style={{ background: "var(--surface-soft)", color: "var(--ink-soft)" }}
                >
                  [{stage.shape.join("×")}]
                </code>
                {changed && (
                  <span
                    className="font-accent rounded-full px-1.5 py-px text-[8px]"
                    style={{ background: `${stage.color}18`, color: stage.color }}
                  >
                    shape ↕
                  </span>
                )}
              </div>
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ─── Probability bars ────────────────────────────────────────────────────────

function ProbBars({ classes, color }: { classes: string[]; color: string }) {
  const raw = classes.map((_, i) => Math.exp(-Math.abs(i - 1) * 0.9));
  const sum = raw.reduce((a, b) => a + b, 0);
  const norm = raw.map((p) => p / sum);
  return (
    <div className="flex flex-col gap-2.5">
      {classes.map((cls, i) => (
        <div key={cls}>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="font-mono text-[10px]" style={{ color: "var(--ink-soft)" }}>
              {cls}
            </span>
            <span className="font-mono text-[11px] font-bold" style={{ color }}>
              {(norm[i] * 100).toFixed(0)}%
            </span>
          </div>
          <div
            className="h-2.5 overflow-hidden rounded-full"
            style={{ background: "var(--surface-soft)" }}
          >
            <motion.div
              className="h-full rounded-full"
              style={{ background: color }}
              initial={{ width: 0 }}
              animate={{ width: `${norm[i] * 100}%` }}
              transition={{ duration: 0.75, delay: i * 0.06, ease: "easeOut" }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Stage inspector (right column) ──────────────────────────────────────────

function TensorDims({ shape, color }: { shape: number[]; color: string }) {
  const labels = ["B", "C", "H", "W"];
  return (
    <div className="flex items-end gap-2">
      {shape.map((v, i) => (
        <div key={i} className="flex flex-col items-center gap-0.5">
          <span className="font-accent text-[8px]" style={{ color: "var(--ink-mute)" }}>
            {labels[i] ?? `d${i}`}
          </span>
          <div
            className="flex items-center justify-center rounded-md border text-[13px] font-bold"
            style={{
              width: v > 999 ? 52 : 36,
              height: 36,
              borderColor: `${color}50`,
              background: `${color}10`,
              color,
              fontFamily: "var(--font-display)",
            }}
          >
            {v}
          </div>
        </div>
      ))}
    </div>
  );
}

function StageInspector({ stage, index }: { stage: Stage; index: number }) {
  const classes = HEAD_CLASSES[stage.id];

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={stage.id}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.22 }}
        className="flex flex-col gap-4 rounded-2xl border p-5"
        style={{ borderColor: `${stage.color}45`, background: `${stage.color}06` }}
      >
        {/* Header */}
        <div className="flex items-start gap-4">
          <HeatmapCanvas seed={stage.heatmapSeed} color={stage.color} size={72} active />
          <div className="flex-1 min-w-0">
            <span
              className="font-accent block text-[9px] uppercase tracking-widest"
              style={{ color: "var(--ink-mute)" }}
            >
              Stage {index + 1} · Inspector
            </span>
            <h3
              className="font-display mt-0.5 text-[17px] font-bold leading-tight"
              style={{ color: stage.color }}
            >
              {stage.label}
            </h3>
            <p
              className="font-mono mt-1.5 text-[10px] leading-snug"
              style={{ color: "var(--ink-soft)" }}
            >
              {stage.note}
            </p>
          </div>
        </div>

        {/* Tensor shape visualizer */}
        <div
          className="rounded-xl border px-4 py-3"
          style={{ borderColor: "var(--border)", background: "var(--surface-soft)" }}
        >
          <div
            className="font-accent mb-2.5 text-[9px] uppercase tracking-widest"
            style={{ color: "var(--ink-mute)" }}
          >
            Output Shape
          </div>
          <TensorDims shape={stage.shape} color={stage.color} />
          <div className="mt-2">
            <code className="font-mono text-[10px]" style={{ color: "var(--ink-soft)" }}>
              [{stage.shape.join(", ")}] · {fmtBytes(stage.shape)}
            </code>
          </div>
        </div>

        {/* Tensor stats table */}
        <div className="overflow-hidden rounded-xl border" style={{ borderColor: "var(--border)" }}>
          <div
            className="border-b px-3 py-2"
            style={{ borderColor: "var(--border)", background: "var(--surface-soft)" }}
          >
            <span
              className="font-accent text-[9px] uppercase tracking-widest"
              style={{ color: "var(--ink-mute)" }}
            >
              Tensor Stats
            </span>
          </div>
          <table className="w-full">
            <tbody>
              {[
                ["dtype", "float32"],
                ["min", (-1.2 * stage.heatmapSeed).toFixed(4)],
                ["max", (3.1 * stage.heatmapSeed).toFixed(4)],
                ["mean", (0.08 * stage.heatmapSeed).toFixed(4)],
                ["std", (0.82 * stage.heatmapSeed).toFixed(4)],
              ].map(([k, v]) => (
                <tr
                  key={k}
                  className="border-b last:border-0"
                  style={{ borderColor: "var(--border)" }}
                >
                  <td
                    className="px-3 py-1.5 font-mono text-[10px]"
                    style={{ color: "var(--ink-mute)", width: "38%" }}
                  >
                    {k}
                  </td>
                  <td
                    className="px-3 py-1.5 text-right font-mono text-[10px] font-semibold"
                    style={{ color: "var(--foreground)" }}
                  >
                    {v}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Predictions */}
        {classes && (
          <div
            className="rounded-xl border p-4"
            style={{ background: "var(--surface-soft)", borderColor: "var(--border)" }}
          >
            <div
              className="font-accent mb-3 text-[9px] uppercase tracking-widest"
              style={{ color: "var(--ink-mute)" }}
            >
              Predicted Probabilities
            </div>
            <ProbBars classes={classes} color={stage.color} />
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}

// ─── Progress ────────────────────────────────────────────────────────────────

function RunProgress({
  current,
  total,
  running,
}: {
  current: number;
  total: number;
  running: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className="h-1.5 flex-1 overflow-hidden rounded-full"
        style={{ background: "var(--surface-soft)" }}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ background: "linear-gradient(90deg,var(--pos),var(--chan),var(--type))" }}
          animate={{ width: `${((current + 1) / total) * 100}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>
      <span className="font-mono shrink-0 text-[10px]" style={{ color: "var(--ink-soft)" }}>
        {current + 1}/{total}
      </span>
      {running && (
        <motion.span
          animate={{ opacity: [1, 0.25, 1] }}
          transition={{ repeat: Infinity, duration: 1 }}
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ background: "var(--pos)" }}
        />
      )}
    </div>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default function DryRunPanel({ backbone }: { backbone: Backbone }) {
  const stages = STAGES(backbone);
  const [activeIndex, setActiveIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopRun = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    setRunning(false);
  }, []);

  const startRun = useCallback(() => {
    setActiveIndex(0);
    setRunning(true);
    let idx = 0;
    intervalRef.current = setInterval(() => {
      idx += 1;
      if (idx >= stages.length) {
        stopRun();
        return;
      }
      setActiveIndex(idx);
    }, 700);
  }, [stages.length, stopRun]);

  useEffect(() => () => stopRun(), [stopRun]);
  useEffect(() => {
    stopRun();
    setActiveIndex(0);
  }, [backbone, stopRun]);

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="mt-10 rounded-3xl border border-dashed border-border bg-gradient-to-b from-surface to-surface-soft p-6 pt-8"
      style={{ position: "relative" }}
    >
      <div
        className="font-accent absolute -top-3 left-6 bg-background px-2.5 text-[11px] uppercase tracking-[0.12em]"
        style={{ color: "var(--ink-soft)" }}
      >
        Dry Run · Image → Predictions
      </div>

      {/* Architecture overview */}
      <ArchOverview activeId={stages[activeIndex].id} stages={stages} />

      {/* Controls */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <button
          onClick={running ? stopRun : startRun}
          className="rounded-full border border-border bg-foreground px-5 py-2 font-display text-xs font-semibold text-background transition-all hover:opacity-80"
        >
          {running ? "⏹ Stop" : "▶ Run Forward Pass"}
        </button>
        <button
          onClick={() => {
            stopRun();
            setActiveIndex(0);
          }}
          className="rounded-full border border-border bg-surface px-4 py-2 font-display text-xs text-foreground transition-all hover:border-foreground"
        >
          ↺ Reset
        </button>
        <div className="min-w-32 flex-1">
          <RunProgress current={activeIndex} total={stages.length} running={running} />
        </div>
      </div>

      {/* Two-column layout */}
      <div className="flex gap-6">
        {/* Left: vertical pipeline rail */}
        <div className="w-48 shrink-0 overflow-y-auto lg:w-52" style={{ maxHeight: 560 }}>
          <PipelineRail
            stages={stages}
            activeIndex={activeIndex}
            onSelect={(i) => {
              stopRun();
              setActiveIndex(i);
            }}
          />
        </div>

        {/* Right: stage inspector */}
        <div className="min-w-0 flex-1">
          <StageInspector stage={stages[activeIndex]} index={activeIndex} />
        </div>
      </div>
    </motion.section>
  );
}
