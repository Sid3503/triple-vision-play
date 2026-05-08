import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Data ─────────────────────────────────────────────────────────────────────

const FOLDS = [
  { id: 1, train: 50, val: 13, acc: 76.9, epoch: 19, cls: "green" as const, badge: "Best fold" },
  { id: 2, train: 50, val: 13, acc: 53.8, epoch: 14, cls: "amber" as const, badge: null },
  { id: 3, train: 50, val: 13, acc: 61.5, epoch: 11, cls: "amber" as const, badge: null },
  { id: 4, train: 51, val: 12, acc: 58.3, epoch: 14, cls: "amber" as const, badge: null },
  { id: 5, train: 51, val: 12, acc: 50.0, epoch: 9,  cls: "red"   as const, badge: "Weakest fold" },
];

const TTA_LABELS = ["orig", "H↔", "V↕", "R+15°", "R−15°", "H+V", "R+H", "R+V"];
const PROB_VALS  = [0.12, 0.22, 0.19, 0.21, 0.14, 0.08, 0.04];
const BSS        = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];

const MIN_ACC = 45, MAX_ACC = 82, RANGE = MAX_ACC - MIN_ACC;
const MEAN    = 60.1;

const ACC_CLR: Record<string, string> = {
  green: "#1D9E75",
  amber: "#C07A0A",
  red:   "#B83820",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function sparklinePath(maxEpoch: number, finalAcc: number) {
  const pts: [string, string][] = [];
  const n = Math.max(maxEpoch, 8);
  for (let i = 0; i <= n; i++) {
    const x = (i / n) * 108 + 6;
    const p = i / n;
    const a =
      28 +
      (finalAcc - 28) * (1 - Math.exp(-4.5 * p)) +
      Math.sin(i * 1.8) * 5 * (1 - p * 0.8);
    const y = 45 - ((Math.max(28, Math.min(finalAcc + 5, a)) - 22) / 62) * 38;
    pts.push([x.toFixed(1), Math.max(4, Math.min(46, y)).toFixed(1)]);
  }
  const d = "M " + pts.map((p) => p.join(",")).join(" L ");
  const last = pts[pts.length - 1];
  return (
    <>
      <path d={d} fill="none" stroke="var(--cv-ft)" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx={last[0]} cy={last[1]} r="2.5" fill="var(--cv-ft)" />
      <text x="6"  y="12" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">acc%</text>
      <text x="6"  y="47" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">1</text>
      <text x={(108 * 0.88 + 6).toFixed(0)} y="47" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">{maxEpoch}</text>
    </>
  );
}

// ─── Heatmap (mirrors DryRunPanel) ────────────────────────────────────────────

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
    data.data[i * 4]     = (r / 255) * v;
    data.data[i * 4 + 1] = (g / 255) * v;
    data.data[i * 4 + 2] = (b / 255) * v;
    data.data[i * 4 + 3] = v * 0.85 + 40;
  }
  ctx.putImageData(data, 0, 0);
}

function HeatmapCanvas({
  seed,
  color,
  size = 44,
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
      className="rounded-lg shrink-0"
      style={{
        imageRendering: "pixelated",
        border: `1.5px solid ${color}50`,
        opacity: active ? 1 : 0.18,
        transition: "opacity 0.4s",
      }}
    />
  );
}

// ─── Pipeline overview strip (mirrors ArchOverview) ───────────────────────────

const PIPELINE_STEPS = [
  { id: "dataset",  label: "Dataset",   sub: "63 images",       color: "var(--cv-frozen)" },
  { id: "folds",    label: "5 Folds",   sub: "StratifiedKFold", color: "var(--cv-ft)" },
  { id: "model",    label: "ResNet50",  sub: "Shared arch",     color: "var(--cv-ft)" },
  { id: "results",  label: "Val Acc",   sub: "Per fold",        color: "var(--cv-head)" },
  { id: "tta",      label: "TTA ×8",    sub: "Augmented",       color: "var(--cv-tta)" },
  { id: "ensemble", label: "Ensemble",  sub: "K-fold avg",      color: "var(--cv-ens)" },
] as const;

function PipelineOverview({ activeId }: { activeId: string }) {
  return (
    <div className="mb-5 flex items-center gap-1 overflow-x-auto pb-1">
      {PIPELINE_STEPS.map((step, si) => {
        const isActive = step.id === activeId;
        return (
          <div key={step.id} className="flex shrink-0 items-center gap-1">
            {si > 0 && (
              <svg width="16" height="12" viewBox="0 0 16 12" fill="none">
                <path
                  d="M1 6 H11 M7 2 L11 6 L7 10"
                  stroke={isActive ? step.color : "var(--border)"}
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
                borderColor: isActive ? step.color : "var(--border)",
                background: isActive ? `${step.color}18` : "var(--surface-soft)",
              }}
            >
              <span
                className="font-display block text-[10px] font-bold leading-none"
                style={{ color: isActive ? step.color : "var(--ink-mute)" }}
              >
                {step.label}
              </span>
              <span
                className="font-accent block text-[8px] leading-none mt-0.5"
                style={{ color: isActive ? step.color : "var(--ink-mute)", opacity: 0.7 }}
              >
                {step.sub}
              </span>
            </motion.div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Section card (mirrors DryRunPanel outer wrapper) ─────────────────────────

function SectionCard({
  label,
  children,
  delay = 0,
}: {
  label: string;
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <motion.div
      className="relative mt-8 rounded-3xl border border-dashed border-border bg-gradient-to-b from-surface to-surface-soft p-5 pt-8"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      style={{ position: "relative" }}
    >
      <div
        className="font-accent absolute -top-3 left-6 bg-background px-2.5 text-[11px] uppercase tracking-[0.12em]"
        style={{ color: "var(--ink-soft)" }}
      >
        {label}
      </div>
      {children}
    </motion.div>
  );
}

// ─── Run progress (mirrors DryRunPanel RunProgress) ────────────────────────────

function RunProgress({
  current,
  total,
  running,
}: {
  current: number;
  total: number;
  running: boolean;
}) {
  const pct = total > 0 ? (current / total) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <div
        className="h-1.5 flex-1 overflow-hidden rounded-full"
        style={{ background: "var(--surface-soft)" }}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ background: "linear-gradient(90deg,var(--cv-frozen),var(--cv-ft),var(--cv-head))" }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.25, ease: "easeOut" }}
        />
      </div>
      <span className="font-mono shrink-0 text-[10px]" style={{ color: "var(--ink-soft)" }}>
        {current}/{total} ep
      </span>
      {running && (
        <motion.span
          animate={{ opacity: [1, 0.25, 1] }}
          transition={{ repeat: Infinity, duration: 1 }}
          className="h-2 w-2 shrink-0 rounded-full"
          style={{ background: "var(--cv-ft)" }}
        />
      )}
    </div>
  );
}

// ─── Connector ────────────────────────────────────────────────────────────────

function Connector({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-3">
      <div className="h-5 w-px shrink-0 rounded" style={{ background: "var(--border)" }} />
      <span
        className="font-accent px-2 text-[9px] uppercase tracking-[0.1em]"
        style={{ color: "var(--ink-mute)", background: "var(--background)" }}
      >
        {label}
      </span>
      <div className="h-5 w-px shrink-0 rounded" style={{ background: "var(--border)" }} />
    </div>
  );
}

// ─── Fold column ──────────────────────────────────────────────────────────────

function FoldColumn({
  fold,
  epoch,
  isHovered,
  isDimmed,
  onEnter,
  onLeave,
}: {
  fold: (typeof FOLDS)[0];
  epoch: number;
  isHovered: boolean;
  isDimmed: boolean;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const total = fold.train + fold.val;
  const trainPct = (fold.train / total) * 100;
  const heatColor = ACC_CLR[fold.cls];

  const borderColor =
    fold.id === 1 ? "var(--cv-ft)" : "var(--border)";

  const accentBg =
    fold.cls === "green"
      ? "var(--cv-ft-bg)"
      : fold.cls === "amber"
        ? "#FFF8EC"
        : "#FDF1EF";

  return (
    <motion.div
      className="relative flex flex-col gap-0 overflow-hidden rounded-2xl border"
      style={{
        borderColor: isHovered ? "var(--foreground)" : borderColor,
        background: isHovered ? `${heatColor}06` : "var(--surface)",
        opacity: isDimmed ? 0.3 : 1,
        transition: "opacity .22s, border-color .22s, box-shadow .22s, background .22s",
        boxShadow:
          fold.id === 1
            ? "0 0 0 1px var(--cv-ft-bd), 0 3px 12px rgba(29,158,117,.09)"
            : isHovered
              ? "0 6px 20px rgba(0,0,0,.09)"
              : "none",
      }}
      animate={{ y: isHovered && !isDimmed ? -3 : 0 }}
      transition={{ type: "spring", stiffness: 320, damping: 28 }}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      {/* top accent line */}
      <div
        className="h-[3px] w-full"
        style={{
          background:
            fold.id === 1
              ? "var(--cv-ft)"
              : fold.id === 5
                ? "#E8B4A0"
                : "var(--bg3,#E8E7E1)",
        }}
      />

      <div className="flex flex-col gap-2 p-3">
        {/* header */}
        <div className="flex items-center justify-between">
          <span className="font-display text-[12px] font-bold">Fold {fold.id}</span>
          {fold.badge && (
            <span
              className="font-accent rounded px-1.5 py-px text-[8px] uppercase tracking-[.04em]"
              style={{
                background: fold.id === 1 ? "var(--cv-ft-bg)" : "var(--surface-soft)",
                color: fold.id === 1 ? "var(--cv-ft)" : "var(--ink-mute)",
              }}
            >
              {fold.badge}
            </span>
          )}
        </div>

        {/* split bar */}
        <div
          className="flex h-2 overflow-hidden rounded-full"
          style={{ background: "var(--surface-soft)" }}
        >
          <div
            className="h-full"
            style={{
              width: `${trainPct}%`,
              background: "linear-gradient(90deg,var(--cv-ft),#55C9A0)",
            }}
          />
          <div
            className="h-full flex-1"
            style={{ background: "linear-gradient(90deg,#F5C350,#E89A20)" }}
          />
        </div>

        {/* counts */}
        <div className="font-mono text-[10px]" style={{ color: "var(--ink-mute)" }}>
          <span style={{ color: "var(--cv-ft)" }}>train: {fold.train}</span>
          <span className="mx-1" style={{ color: "var(--border)" }}>|</span>
          <span style={{ color: "#C07A0A" }}>val: {fold.val}</span>
        </div>

        {/* epoch counter */}
        <div className="text-center">
          <div className="font-display text-[22px] font-bold leading-none">{epoch}</div>
          <div
            className="font-accent mt-0.5 text-[8px] uppercase tracking-[.08em]"
            style={{ color: "var(--ink-mute)" }}
          >
            / {fold.epoch} epochs
          </div>
        </div>

        {/* progress bar */}
        <div className="h-[3px] overflow-hidden rounded-full" style={{ background: "var(--surface-soft)" }}>
          <motion.div
            className="h-full rounded-full"
            style={{ background: "linear-gradient(90deg,var(--cv-ft),#55C9A0)" }}
            animate={{ width: `${epoch === 0 ? 0 : (epoch / fold.epoch) * 100}%` }}
            transition={{ duration: 0.06 }}
          />
        </div>

        {/* hover detail */}
        <AnimatePresence>
          {isHovered && (
            <motion.div
              key="detail"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.28 }}
              className="overflow-hidden"
            >
              <div className="mb-2 mt-1 h-px" style={{ background: "var(--border)" }} />

              {/* heatmap + accuracy (mirrors StageInspector header layout) */}
              <div className="flex items-start gap-2 mb-2">
                <HeatmapCanvas
                  seed={fold.acc / 100}
                  color={heatColor}
                  size={44}
                  active={isHovered}
                />
                <div className="flex-1 min-w-0 text-center">
                  <div
                    className="font-display text-[24px] font-bold leading-none"
                    style={{ color: ACC_CLR[fold.cls] }}
                  >
                    {fold.acc}%
                  </div>
                  <div
                    className="font-accent mt-1 text-[8px] uppercase tracking-[.06em]"
                    style={{ color: "var(--ink-mute)" }}
                  >
                    val acc · ep {fold.epoch}
                  </div>
                </div>
              </div>

              {/* sparkline */}
              <div className="rounded-xl border p-1.5" style={{ background: accentBg, borderColor: `${heatColor}30` }}>
                <svg viewBox="0 0 120 52" className="h-10 w-full" preserveAspectRatio="xMidYMid meet">
                  {sparklinePath(fold.epoch, fold.acc)}
                </svg>
              </div>
              <div
                className="font-mono mt-1.5 text-center text-[9px]"
                style={{ color: "var(--ink-mute)" }}
              >
                patience=12 · early stopped
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// ─── Model architecture ────────────────────────────────────────────────────────

type PulseTarget = "none" | "ft" | "head";

function ModelArch({
  freezeOn,
  pulse,
}: {
  freezeOn: boolean;
  pulse: PulseTarget;
}) {
  const frozenChipStyle = {
    background: freezeOn ? undefined : "var(--cv-frozen-bg)",
    borderColor: "var(--cv-frozen-bd)",
    color: "var(--cv-frozen)",
  };

  const ftClass  = pulse === "ft"   ? "cv-pulse-ft" : "";
  const hdClass  = pulse === "head" ? "cv-pulse-hd" : "";

  const Chip = ({
    label,
    sub,
    kind,
    extraClass = "",
  }: {
    label: string;
    sub?: string;
    kind: "frozen" | "ft" | "head" | "head-out";
    extraClass?: string;
  }) => {
    const base =
      "inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] font-semibold transition-all duration-300 ";
    const styles: Record<string, React.CSSProperties> = {
      frozen:     frozenChipStyle,
      ft:         { background: "var(--cv-ft-bg)",   borderColor: "var(--cv-ft-bd)",   color: "var(--cv-ft)" },
      head:       { background: "var(--cv-head-bg)", borderColor: "var(--cv-head-bd)", color: "var(--cv-head)" },
      "head-out": { background: "var(--cv-head)",    borderColor: "var(--cv-head)",    color: "white" },
    };
    return (
      <div
        className={`${base} ${kind === "frozen" && freezeOn ? "cv-frozen-hatch" : ""} ${extraClass}`}
        style={styles[kind]}
      >
        {kind === "frozen" && freezeOn && <span className="text-[9px]">🔒</span>}
        {label}
        {sub && <span className="ml-1 text-[8px] opacity-60">{sub}</span>}
      </div>
    );
  };

  const Arrow = () => (
    <span className="text-[11px]" style={{ color: "var(--cv-head-bd)" }}>→</span>
  );

  return (
    <div>
      {/* header pills */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {[
          "ResNet50 pretrained ImageNet",
          "23.2M trainable params",
          "AdamW · weight_decay=1e-4",
          "CrossEntropyLoss · label_smoothing=0.1",
        ].map((t) => (
          <span
            key={t}
            className="rounded-md border px-2 py-1 font-mono text-[11px]"
            style={{ borderColor: "var(--border)", color: "var(--ink-soft)" }}
          >
            {t}
          </span>
        ))}
      </div>

      {/* frozen row */}
      <div className="mb-3 grid grid-cols-[72px_1fr] items-start gap-3">
        <div className="pt-2 font-mono text-[10px] font-bold uppercase tracking-[.06em]" style={{ color: "var(--cv-frozen)" }}>
          🔒 Frozen
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Chip label="conv1" kind="frozen" />
          <Chip label="layer1" sub="64ch" kind="frozen" />
          <Chip label="layer2" sub="128ch" kind="frozen" />
          <span className="font-mono text-[10px]" style={{ color: "var(--ink-mute)" }}>
            Fixed ImageNet extractors · lr = 0
          </span>
        </div>
      </div>

      {/* fine-tuned row */}
      <div className="mb-3 grid grid-cols-[72px_1fr] items-start gap-3">
        <div className="pt-2 font-mono text-[10px] font-bold uppercase tracking-[.06em]" style={{ color: "var(--cv-ft)" }}>
          ⚡ Fine-tuned
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Chip label="layer3" sub="256ch" kind="ft" extraClass={ftClass} />
          <Chip label="layer4" sub="512ch" kind="ft" extraClass={ftClass} />
          <span className="font-mono text-[10px]" style={{ color: "var(--ink-mute)" }}>
            AdamW · lr = 1e-4 · CosineAnnealingWarmRestarts (T₀=10, T_mult=2)
          </span>
        </div>
      </div>

      {/* head row */}
      <div className="mb-4 grid grid-cols-[72px_1fr] items-start gap-3">
        <div className="pt-2 font-mono text-[10px] font-bold uppercase tracking-[.06em]" style={{ color: "var(--cv-head)" }}>
          ★ Head
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {[
            ["GAP", "2048"],
            ["Drop", "0.5"],
            ["Lin", "→512"],
            ["ReLU+BN", ""],
            ["Drop", "0.3"],
            ["Lin", "→256"],
            ["ReLU+BN", ""],
            ["Drop", "0.2"],
          ].map(([l, s], i) => (
            <div key={i} className="flex items-center gap-1">
              <Chip label={l} sub={s || undefined} kind="head" extraClass={hdClass} />
              <Arrow />
            </div>
          ))}
          <Chip label="Lin" sub="→7" kind="head-out" extraClass={hdClass} />
          <span className="font-mono text-[10px]" style={{ color: "var(--ink-mute)" }}>
            lr = 1e-3 · softmax
          </span>
        </div>
      </div>

      {/* training strategy pills */}
      <div className="flex flex-wrap gap-1.5 border-t pt-3" style={{ borderColor: "var(--border)" }}>
        <span
          className="font-accent self-center text-[9px] uppercase tracking-[.08em]"
          style={{ color: "var(--ink-mute)" }}
        >
          Strategy:
        </span>
        {[
          ["🎨 MixUp α=0.3",        "#FFF3E0","#FFCC80","#EF9F27"],
          ["✂ CutMix p=0.5",        "#FDF0EA","#F0B090","#D85A30"],
          ["〜 EMA decay=0.999",     "#EBF3FD","#A8CCF2","#378ADD"],
          ["📊 SWA ep.15/3",         "#EBF3FD","#A8CCF2","#378ADD"],
          ["📉 CosineWarmRestart",   "#F4EBF9","#D4A8F0","#8B40C4"],
          ["∇ Grad clip=1.0",        "var(--surface-soft)","var(--border)","var(--ink-soft)"],
          ["⏹ Early stop p=12",      "var(--surface-soft)","var(--border)","var(--ink-soft)"],
          ["⚖ LabelSmooth=0.1",      "var(--surface-soft)","var(--border)","var(--ink-soft)"],
        ].map(([lbl, bg, bd, clr]) => (
          <span
            key={lbl}
            className="rounded-md border px-2 py-1 font-mono text-[10px] font-medium"
            style={{ background: bg, borderColor: bd, color: clr }}
          >
            {lbl}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Accuracy bars ─────────────────────────────────────────────────────────────

function AccuracyBars({ visible }: { visible: boolean }) {
  const meanPos = ((MEAN - MIN_ACC) / RANGE) * 100;

  return (
    <div>
      <div className="mb-4 font-mono text-[11px]" style={{ color: "var(--ink-soft)" }}>
        Ensemble mean:{" "}
        <strong style={{ color: "var(--foreground)" }}>60.1% ± 9.3%</strong>
        &nbsp;·&nbsp;bar scale 45–82%&nbsp;·&nbsp;
        <span style={{ borderBottom: "1.5px dashed #888" }}>dashed = ensemble mean</span>
      </div>

      <div className="relative pt-5">
        {FOLDS.map((f, fi) => {
          const fillPct = ((f.acc - MIN_ACC) / RANGE) * 100;
          return (
            <div key={f.id} className="mb-2 flex items-center gap-3">
              <span
                className="w-14 shrink-0 font-display text-[11px] font-semibold"
                style={{ color: "var(--ink-soft)" }}
              >
                {f.id === 1 ? "★ " : f.id === 5 ? "↓ " : ""}Fold {f.id}
              </span>

              <div
                className="relative h-6 flex-1 overflow-visible rounded-md"
                style={{ background: "var(--surface-soft)" }}
              >
                {fi === 0 && (
                  <div
                    className="absolute top-0 h-full"
                    style={{ left: `${meanPos}%`, borderLeft: "1.5px dashed rgba(80,80,80,.4)" }}
                  >
                    <span
                      className="font-mono absolute -top-5 left-1 whitespace-nowrap text-[9px]"
                      style={{ color: "#888" }}
                    >
                      mean {MEAN}%
                    </span>
                  </div>
                )}
                {fi > 0 && (
                  <div
                    className="pointer-events-none absolute top-0 h-full"
                    style={{ left: `${meanPos}%`, borderLeft: "1.5px dashed rgba(80,80,80,.28)" }}
                  />
                )}

                <motion.div
                  className="flex h-full items-center justify-end rounded-md pr-2"
                  style={{ background: ACC_CLR[f.cls] }}
                  animate={{ width: visible ? `${fillPct}%` : "0%" }}
                  transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1], delay: fi * 0.08 }}
                >
                  <span className="font-display text-[10px] font-bold text-white">
                    {f.acc}%
                  </span>
                </motion.div>
              </div>

              <span
                className="w-24 shrink-0 font-mono text-[10px]"
                style={{ color: "var(--ink-mute)" }}
              >
                ep.{f.epoch} · {f.train}+{f.val}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── TTA grid ─────────────────────────────────────────────────────────────────

function TTAGrid({ litItems }: { litItems: Set<string> }) {
  return (
    <div>
      <div className="mb-4 font-mono text-[11px]" style={{ color: "var(--ink-soft)" }}>
        Each fold model runs 8 augmented passes → average softmax → 1 probability vector per fold
      </div>

      <div className="grid grid-cols-5 gap-2.5">
        {FOLDS.map((f) => (
          <div
            key={f.id}
            className="rounded-xl border p-2.5 text-center"
            style={{ borderColor: "var(--border)", background: "var(--surface)" }}
          >
            <div
              className="font-display mb-2 text-[10px] font-semibold"
              style={{ color: "var(--ink-soft)" }}
            >
              Fold {f.id}
            </div>
            <div className="mb-2 flex flex-wrap justify-center gap-1">
              {TTA_LABELS.map((lbl, i) => {
                const key = `${f.id}_${i}`;
                const lit = litItems.has(key);
                return (
                  <motion.div
                    key={i}
                    className="flex h-5 w-5 items-center justify-center rounded font-mono text-[7px] font-bold"
                    style={{
                      background: lit ? "var(--cv-tta)" : "var(--cv-tta-bg)",
                      border: `1px solid ${lit ? "var(--cv-tta)" : "var(--cv-tta-bd)"}`,
                      color: lit ? "white" : "var(--cv-tta)",
                    }}
                    animate={{ scale: lit ? 1.15 : 1 }}
                    title={lbl}
                  >
                    {i + 1}
                  </motion.div>
                );
              })}
            </div>
            <div className="font-accent mb-1 text-[9px]" style={{ color: "var(--ink-mute)" }}>
              ↓ avg 8 vectors
            </div>
            <div
              className="inline-block rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold"
              style={{
                background: "var(--cv-tta-bg)",
                border: "1px solid var(--cv-tta-bd)",
                color: "var(--cv-tta)",
              }}
            >
              [1×7] softmax
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-2">
        <div className="h-px flex-1" style={{ background: "var(--border)" }} />
        <span className="font-accent px-2 text-[9px] uppercase tracking-[.08em]" style={{ color: "var(--ink-mute)" }}>
          average 5 fold vectors with equal weight
        </span>
        <div className="h-px flex-1" style={{ background: "var(--border)" }} />
      </div>
    </div>
  );
}

// ─── Ensemble output ───────────────────────────────────────────────────────────

function EnsembleOutput({ probsVisible }: { probsVisible: boolean }) {
  return (
    <div className="flex w-full flex-col items-center gap-3">
      {/* K-fold output node */}
      <div
        className="relative w-full max-w-xl overflow-hidden rounded-2xl p-6 text-center"
        style={{ background: "var(--cv-ens)", color: "var(--cv-ens-t)" }}
      >
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse at 20% 50%,rgba(45,201,148,.14) 0%,transparent 60%)," +
              "radial-gradient(ellipse at 80% 50%,rgba(127,119,221,.1) 0%,transparent 60%)",
          }}
        />
        <div className="font-accent mb-2 text-[9px] uppercase tracking-[.1em] opacity-60">
          K-Fold Ensemble · ResNet50
        </div>
        <div className="font-display mb-1 text-[14px] font-bold">
          7-class BSS type probability vector &nbsp;[1 × 7]
        </div>
        <div className="font-mono mb-4 text-[10px] opacity-60">
          60.1% mean val acc · ±9.3% std · 5 independent fold models
        </div>

        {/* mini prob chart */}
        <div className="flex items-end justify-center gap-1.5 pb-5">
          {PROB_VALS.map((v, i) => (
            <div key={i} className="flex flex-col items-center gap-1">
              <div
                className="relative w-8 h-11 overflow-hidden rounded-t-md"
                style={{ background: "rgba(255,255,255,.1)" }}
              >
                <motion.div
                  className="absolute bottom-0 w-full rounded-t-md"
                  style={{ background: "linear-gradient(to top,#2DC994,#A0EACF)" }}
                  animate={{ height: probsVisible ? `${v * 100 * 2.5}%` : "0%" }}
                  transition={{ duration: 1.3, ease: [0.22, 1, 0.36, 1], delay: i * 0.07 }}
                />
              </div>
              <span className="font-accent text-[8px] opacity-65">{BSS[i]}</span>
            </div>
          ))}
        </div>
      </div>

      <Connector label="1 of 7 inputs into the final model ensemble" />

      {/* Final 7-model box */}
      <div
        className="w-full max-w-xl rounded-2xl border p-5 text-center"
        style={{
          background: "var(--surface-soft)",
          borderStyle: "dashed",
          borderColor: "var(--border)",
        }}
      >
        <div className="font-display mb-1.5 text-[12px] font-semibold" style={{ color: "var(--ink-soft)" }}>
          Final 7-Model Ensemble Average
        </div>
        <div className="font-mono mb-3 text-[11px]" style={{ color: "var(--ink-mute)" }}>
          Equal-weight average of 7 probability vectors → argmax → BSS type
        </div>
        <div className="flex flex-wrap justify-center gap-2">
          {[
            ["ResNet50 K-Fold", true],
            ["StoolNetTriple · Type head", false],
            ["StoolNetTriple · Shape head", false],
            ["StoolNetTriple · Color head", false],
            ["+ 3 additional models", false],
          ].map(([lbl, here]) => (
            <span
              key={lbl as string}
              className="rounded-md border px-2.5 py-1 font-mono text-[10px]"
              style={
                here
                  ? { background: "var(--cv-ens)", color: "var(--cv-ens-t)", borderColor: "var(--cv-ens)" }
                  : { background: "white", color: "var(--ink-soft)", borderColor: "var(--border)" }
              }
            >
              {lbl as string}
              {here ? " ← this pipeline" : ""}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Variance modal ────────────────────────────────────────────────────────────

function VarianceModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-foreground/25 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ type: "spring", stiffness: 340, damping: 32 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            onClick={onClose}
          >
            <div
              className="relative max-w-md rounded-2xl bg-card p-7 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={onClose}
                className="absolute right-4 top-4 flex h-7 w-7 items-center justify-center rounded-lg border font-mono text-[12px]"
                style={{ borderColor: "var(--border)", color: "var(--ink-mute)" }}
              >
                ✕
              </button>
              <h3 className="font-display mb-3 text-[15px] font-bold">
                📊 Why such high fold variance?
              </h3>
              <div
                className="space-y-3 font-mono text-[12px] leading-relaxed"
                style={{ color: "var(--ink-soft)" }}
              >
                <p>
                  With only{" "}
                  <strong style={{ color: "var(--foreground)" }}>12–13 validation images per fold</strong>,
                  a single misclassified image changes accuracy by{" "}
                  <strong style={{ color: "var(--foreground)" }}>7–8 percentage points</strong>.
                </p>
                <p>
                  Both Fold 1 (76.9%) and Fold 5 (50.0%) represent valid training runs —
                  the spread is unavoidable with a 63-image dataset, not a training defect.
                </p>
                <p>
                  Fold 1 getting ~3 more images correct than Fold 5 fully explains the 26.9 pp gap.
                </p>
                <div
                  className="rounded-lg border-l-2 py-3 pl-4 font-mono text-[11px] leading-6"
                  style={{
                    background: "var(--cv-ft-bg)",
                    borderColor: "var(--cv-ft)",
                    color: "var(--foreground)",
                  }}
                >
                  ✓ &nbsp;1 image error in val13 = 1/13 ≈ 7.7 pp<br />
                  ✓ &nbsp;Fold 1→5 gap = ~3–4 images = 26.9 pp<br />
                  ✓ &nbsp;Ensemble (60.1%) is more stable than any single fold<br />
                  ✓ &nbsp;Always report the ensemble, never cherry-pick
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

// ─── Dataset row ──────────────────────────────────────────────────────────────

function DatasetRow() {
  return (
    <div className="flex items-center gap-5">
      {/* thumbnail grid */}
      <div className="grid shrink-0 gap-[3px]" style={{ gridTemplateColumns: "repeat(9,1fr)" }}>
        {Array.from({ length: 63 }).map((_, i) => {
          const hues = ["#D4C4A4","#C6B592","#B9A880","#AD9C70","#A19060","#958452","#8A7848"];
          return (
            <div
              key={i}
              className="h-3.5 w-3.5 rounded-[2px]"
              style={{ background: hues[i % 7] }}
            />
          );
        })}
      </div>

      <div className="flex-1">
        <div className="font-display mb-1.5 text-[16px] font-bold">BSS Stool Image Dataset</div>
        <div className="space-y-0.5 font-mono text-[11px]" style={{ color: "var(--ink-soft)" }}>
          <div>63 original images · pre-augmentation</div>
          <div>7 BSS types · ImageNet-normalised · 224×224 px</div>
          <div>~9 images per class · class-weighted CrossEntropyLoss</div>
        </div>
      </div>

      <div
        className="shrink-0 rounded-xl border px-3 py-2.5 font-mono text-[10px] leading-6"
        style={{
          background: "var(--surface-soft)",
          borderColor: "var(--border)",
          color: "var(--ink-soft)",
        }}
      >
        <strong style={{ color: "var(--foreground)" }}>StratifiedKFold</strong>
        <br />n_splits = 5<br />shuffle = True<br />random_state = 42
      </div>
    </div>
  );
}

// ─── Legend ────────────────────────────────────────────────────────────────────

function Legend() {
  const items: [string, string, string][] = [
    ["Frozen (ImageNet)", "var(--cv-frozen-bg)", "var(--cv-frozen-bd)"],
    ["Fine-tuned", "var(--cv-ft-bg)", "var(--cv-ft-bd)"],
    ["Classification head", "var(--cv-head-bg)", "var(--cv-head-bd)"],
    ["MixUp", "#FFF3E0", "#FFCC80"],
    ["CutMix", "#FDF0EA", "#F0B090"],
    ["EMA / SWA", "#EBF3FD", "#A8CCF2"],
    ["TTA ×8", "var(--cv-tta-bg)", "var(--cv-tta-bd)"],
    ["Ensemble output", "var(--cv-ens)", "var(--cv-ens)"],
  ];
  return (
    <div className="mb-5 flex flex-wrap gap-3">
      {items.map(([lbl, bg, bd]) => (
        <div key={lbl} className="flex items-center gap-1.5 font-mono text-[11px]" style={{ color: "var(--ink-soft)" }}>
          <div
            className="h-2.5 w-2.5 shrink-0 rounded-[3px]"
            style={{ background: bg, border: `1.5px solid ${bd}` }}
          />
          {lbl}
        </div>
      ))}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function ResNetCVPipeline() {
  const [simRunning, setSimRunning]     = useState(false);
  const [epochCounts, setEpochCounts]   = useState<Record<number, number>>({});
  const [freezeOn, setFreezeOn]         = useState(false);
  const [ttaRunning, setTtaRunning]     = useState(false);
  const [litTTA, setLitTTA]             = useState<Set<string>>(new Set());
  const [varianceOpen, setVarianceOpen] = useState(false);
  const [hoveredFold, setHoveredFold]   = useState<number | null>(null);
  const [barsVisible, setBarsVisible]   = useState(false);
  const [probsVisible, setProbsVisible] = useState(false);
  const [pulse, setPulse]               = useState<PulseTarget>("none");

  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    const t1 = setTimeout(() => setBarsVisible(true), 500);
    const t2 = setTimeout(() => setProbsVisible(true), 900);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const stopSim = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setSimRunning(false);
    setEpochCounts({});
    setPulse("none");
  }, []);

  const startSim = useCallback(() => {
    if (simRunning) { stopSim(); return; }
    setSimRunning(true);
    setEpochCounts({});

    FOLDS.forEach((f, fi) => {
      const base = fi * 1300;

      timers.current.push(setTimeout(() => { setPulse("none"); }, base));

      timers.current.push(
        setTimeout(() => {
          setPulse("ft");
          timers.current.push(setTimeout(() => setPulse("none"), 1900));
        }, base + 250),
      );

      timers.current.push(
        setTimeout(() => {
          setPulse("head");
          timers.current.push(setTimeout(() => setPulse("none"), 1900));
        }, base + 600),
      );

      function tick(ep: number) {
        setEpochCounts((prev) => ({ ...prev, [f.id]: ep }));
        if (ep < f.epoch && simRunning !== false) {
          const t = setTimeout(() => tick(ep + 1), 55);
          timers.current.push(t);
        }
      }
      timers.current.push(setTimeout(() => tick(1), base + 120));
    });

    const total = FOLDS.reduce(
      (mx, f, fi) => Math.max(mx, fi * 1300 + f.epoch * 55 + 900),
      0,
    );
    timers.current.push(
      setTimeout(() => {
        setSimRunning(false);
        setPulse("none");
      }, total),
    );
  }, [simRunning, stopSim]);

  const runTTA = useCallback(() => {
    if (ttaRunning) return;
    setTtaRunning(true);
    setLitTTA(new Set());

    FOLDS.forEach((f, fi) => {
      const base = fi * 280;
      for (let i = 0; i < 8; i++) {
        const key = `${f.id}_${i}`;
        timers.current.push(
          setTimeout(() => {
            setLitTTA((prev) => new Set([...prev, key]));
            timers.current.push(
              setTimeout(() => {
                setLitTTA((prev) => {
                  const next = new Set(prev);
                  next.delete(key);
                  return next;
                });
              }, 480),
            );
          }, base + i * 110),
        );
      }
    });

    const total = FOLDS.length * 280 + 8 * 110 + 600;
    timers.current.push(setTimeout(() => setTtaRunning(false), total));
  }, [ttaRunning]);

  useEffect(() => {
    const t = setTimeout(() => startSim(), 700);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // derive pipeline overview active step
  const activeOverviewId =
    simRunning
      ? "folds"
      : ttaRunning
        ? "tta"
        : probsVisible
          ? "ensemble"
          : barsVisible
            ? "results"
            : Object.keys(epochCounts).length > 0
              ? "model"
              : "dataset";

  // overall training progress
  const totalEpochs = FOLDS.reduce((s, f) => s + f.epoch, 0);
  const doneEpochs  = Object.values(epochCounts).reduce((s, v) => s + (v as number), 0);

  return (
    <div className="mx-auto max-w-[1060px] px-5 pb-20 pt-6">
      {/* controls — matches DryRunPanel button style */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.05 }}
        className="sticky top-0 z-30 mb-6 rounded-3xl border border-dashed border-border bg-gradient-to-b from-surface to-surface-soft px-5 py-4 backdrop-blur-sm"
        style={{ position: "relative" }}
      >
        <div
          className="font-accent absolute -top-3 left-6 bg-background px-2.5 text-[11px] uppercase tracking-[0.12em]"
          style={{ color: "var(--ink-soft)" }}
        >
          Training Controls
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={simRunning ? stopSim : startSim}
            className="rounded-full border border-border bg-foreground px-5 py-2 font-display text-xs font-semibold text-background transition-all hover:opacity-80"
          >
            {simRunning ? "⏹ Stop" : "▶ Simulate Training"}
          </button>

          <button
            onClick={() => setFreezeOn((v) => !v)}
            className="rounded-full border border-border bg-surface px-4 py-2 font-display text-xs text-foreground transition-all hover:border-foreground"
            style={freezeOn ? { borderColor: "var(--cv-ft)", color: "var(--cv-ft)" } : {}}
          >
            🔒 {freezeOn ? "Hide" : "Show"} frozen layers
          </button>

          <button
            disabled={ttaRunning}
            onClick={runTTA}
            className="rounded-full border border-border bg-surface px-4 py-2 font-display text-xs text-foreground transition-all hover:border-foreground disabled:opacity-50"
          >
            ⚡ {ttaRunning ? "Running…" : "Run TTA ×8"}
          </button>

          <button
            onClick={() => setVarianceOpen(true)}
            className="rounded-full border border-border bg-surface px-4 py-2 font-display text-xs text-foreground transition-all hover:border-foreground"
          >
            ? Why high variance?
          </button>

          <div className="min-w-32 flex-1">
            <RunProgress
              current={doneEpochs}
              total={totalEpochs}
              running={simRunning}
            />
          </div>
        </div>
      </motion.section>

      {/* page title */}
      <motion.div
        className="mb-5"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08 }}
      >
        <h1 className="font-display mb-1 text-[19px] font-bold tracking-tight">
          ResNet50 · 5-Fold Cross-Validation Training &amp; Inference Pipeline
        </h1>
        <p className="font-mono text-[11px]" style={{ color: "var(--ink-mute)" }}>
          63 images · 7 BSS classes · StratifiedKFold · TTA ×8 · K-Fold ensemble
        </p>
      </motion.div>

      {/* pipeline overview strip (mirrors ArchOverview) */}
      <PipelineOverview activeId={activeOverviewId} />

      <Legend />

      {/* 01 Dataset */}
      <SectionCard label="01 · Dataset" delay={0.05}>
        <DatasetRow />
      </SectionCard>

      <Connector label="stratified split · proportional class distribution per fold" />

      {/* 02 Folds */}
      <SectionCard label="02 · Five Stratified Folds" delay={0.12}>
        <div className="grid grid-cols-5 gap-2.5">
          {FOLDS.map((f) => (
            <FoldColumn
              key={f.id}
              fold={f}
              epoch={epochCounts[f.id] ?? 0}
              isHovered={hoveredFold === f.id}
              isDimmed={hoveredFold !== null && hoveredFold !== f.id}
              onEnter={() => setHoveredFold(f.id)}
              onLeave={() => setHoveredFold(null)}
            />
          ))}
        </div>
      </SectionCard>

      <Connector label="each fold trains an independent copy of the shared architecture" />

      {/* 03 Model */}
      <SectionCard label="03 · Shared Model Architecture — ResNet50" delay={0.2}>
        <ModelArch freezeOn={freezeOn} pulse={pulse} />
      </SectionCard>

      <Connector label="per-fold validation results after early stopping" />

      {/* 04 Results */}
      <SectionCard label="04 · Training Results · Val Accuracy per Fold" delay={0.27}>
        <AccuracyBars visible={barsVisible} />
      </SectionCard>

      <Connector label="inference · test-time augmentation × 8 per fold" />

      {/* 05 Inference */}
      <SectionCard label="05 · Inference Pipeline · TTA ×8" delay={0.34}>
        <TTAGrid litItems={litTTA} />
      </SectionCard>

      <Connector label="k-fold ensemble output" />

      {/* 06 Ensemble */}
      <SectionCard label="06 · K-Fold Ensemble Output" delay={0.41}>
        <div className="flex justify-center">
          <EnsembleOutput probsVisible={probsVisible} />
        </div>
      </SectionCard>

      <VarianceModal open={varianceOpen} onClose={() => setVarianceOpen(false)} />
    </div>
  );
}
