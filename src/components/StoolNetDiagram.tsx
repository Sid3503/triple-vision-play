import { useEffect, useLayoutEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence, LayoutGroup } from "framer-motion";

type Backbone = "dense" | "resnet";
type BranchKey = "pos" | "chan" | "type";

const BACKBONE = {
  dense: { name: "DenseNet121", dim: 1024 },
  resnet: { name: "ResNet50", dim: 2048 },
};

const BRANCHES: {
  key: BranchKey;
  index: string;
  title: string;
  sub: string;
  steps: string[];
  formula: string;
  tip: string;
}[] = [
  {
    key: "pos",
    index: "Branch 1",
    title: "Position Attention",
    sub: "Spatial self-attention",
    steps: [
      "Conv2d × 3 → Q, K, V projections",
      "Softmax(Q × Kᵀ) → spatial attention matrix",
      "Apply attention to V → attended output",
      "Residual: γ · attn + x  (γ init 0.1)",
    ],
    formula: "Softmax(QKᵀ / √d) · V",
    tip: "Spatial self-attention. Captures long-range dependencies between pixel positions.",
  },
  {
    key: "chan",
    index: "Branch 2",
    title: "Channel Attention",
    sub: "Inter-channel correlation",
    steps: [
      "Reshape feature map → C × (H·W)",
      "Softmax(C × Cᵀ) → C×C correlation matrix",
      "Re-weight channels by co-occurrence",
      "Residual: γ · attn + x  (γ init 0.1)",
    ],
    formula: "A_c = Softmax(C · Cᵀ)",
    tip: "Inter-channel correlation. Models how feature channels co-occur and re-weights them.",
  },
  {
    key: "type",
    index: "Branch 3",
    title: "Type Attention",
    sub: "Texture · ordinal continuity",
    steps: [
      "Multi-scale Conv (3×3, 5×5, 7×7) ∥",
      "Scale Attention (SE squeeze-excite)",
      "Depthwise-separable texture analyser",
      "Sigmoid continuity gate (BSS 1–7)",
      "Two-layer Type Enhancer network",
      "Residual: γ · attn + x",
    ],
    formula: "MS = Σ SE(Conv_k(x)) · σ(gate)",
    tip: "Texture + ordinal continuity. Multi-scale convs and a sigmoid gate enforce ordering across BSS Types 1–7.",
  },
];

const HEADS = [
  {
    key: "type",
    title: "Head 1 · BSS Type",
    arch: "512 → 256 → 7",
    loss: "FocalLoss (γ=2.0, ls=0.1) × 2.0",
    lossShort: "FocalLoss · ×2.0",
    classes: "Bristol Stool Types 1–7 + confidence",
    color: "var(--head-type)",
    tip: "Bristol Stool Scale type classification (Types 1–7) with confidence.",
  },
  {
    key: "shape",
    title: "Head 2 · Stool Shape",
    arch: "256 → 4",
    loss: "CrossEntropy × 0.8",
    lossShort: "CrossEntropy · ×0.8",
    classes: "Constipation · Normal · MildDiarrhea · Liquid",
    color: "var(--head-shape)",
    tip: "Coarse stool shape classification — clinical proxy.",
  },
  {
    key: "color",
    title: "Head 3 · Stool Color",
    arch: "128 → 2",
    loss: "CrossEntropy × 0.1",
    lossShort: "CrossEntropy · ×0.1",
    classes: "Brown · Yellow",
    color: "var(--head-color)",
    tip: "Auxiliary colour classification, low loss weight.",
  },
];

// ---------- Connector layer ----------
type PathInfo = {
  d: string;
  kind: "main" | BranchKey | "head";
  delay: number;
  length: number;
};

function useConnectors(
  containerRef: React.RefObject<HTMLDivElement | null>,
  refs: Record<string, React.RefObject<HTMLElement | null>>,
  branchRefs: React.RefObject<Record<BranchKey, HTMLElement | null>>,
  headRefs: React.RefObject<HTMLElement[]>,
  deps: unknown[],
) {
  const [paths, setPaths] = useState<PathInfo[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const compute = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const c = container.getBoundingClientRect();
    setSize({ w: c.width, h: c.height });

    const center = (el: HTMLElement | null) => {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return {
        x: r.left - c.left + r.width / 2,
        top: r.top - c.top,
        bottom: r.bottom - c.top,
      };
    };
    const input = center(refs.input.current);
    const backbone = center(refs.backbone.current);
    const triple = center(refs.triple.current);
    const fusion = center(refs.fusion.current);
    const gap = center(refs.gap.current);
    if (!input || !backbone || !triple || !fusion || !gap) return;

    const vline = (x1: number, y1: number, x2: number, y2: number) => {
      const my = (y1 + y2) / 2;
      return `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
    };

    const measure = (d: string) => {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", d);
      // approximate length via temporary SVG
      const svgNs = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svgNs.appendChild(p);
      document.body.appendChild(svgNs);
      const len = p.getTotalLength();
      document.body.removeChild(svgNs);
      return len;
    };

    const out: PathInfo[] = [];
    const push = (d: string, kind: PathInfo["kind"], delay: number) =>
      out.push({ d, kind, delay, length: measure(d) });

    push(vline(input.x, input.bottom, backbone.x, backbone.top), "main", 0);
    push(vline(backbone.x, backbone.bottom, triple.x, triple.top), "main", 0.15);

    (Object.keys(branchRefs.current ?? {}) as BranchKey[]).forEach((k, i) => {
      const b = center(branchRefs.current?.[k] ?? null);
      if (!b) return;
      push(vline(triple.x, triple.top + 6, b.x, b.top), k, 0.3 + i * 0.08);
      push(vline(b.x, b.bottom, fusion.x, fusion.top), k, 0.55 + i * 0.08);
    });

    push(vline(fusion.x, fusion.bottom, gap.x, gap.top), "main", 0.85);

    headRefs.current?.forEach((h) => {
      const hc = center(h);
      if (!hc) return;
      push(vline(gap.x, gap.bottom, hc.x, hc.top), "head", 0.95);
    });

    setPaths(out);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerRef, refs, branchRefs, headRefs]);

  useLayoutEffect(() => {
    compute();
    const ro = new ResizeObserver(() => compute());
    if (containerRef.current) ro.observe(containerRef.current);
    window.addEventListener("resize", compute);
    // refresh after fonts settle
    const t1 = setTimeout(compute, 300);
    const t2 = setTimeout(compute, 1200);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", compute);
      clearTimeout(t1);
      clearTimeout(t2);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { paths, size };
}

// ---------- Animated pulse along path ----------
function FlowPulse({
  d,
  length,
  delay,
  color,
}: {
  d: string;
  length: number;
  delay: number;
  color: string;
}) {
  // Use SVG <animateMotion>-free approach with framer-motion offsetDistance via stroke
  // We'll animate a circle with CSS via getPointAtLength sampled in JS.
  const pathRef = useRef<SVGPathElement>(null);
  const dotRef = useRef<SVGCircleElement>(null);
  const startRef = useRef<number | null>(null);
  const DURATION = 4200;

  useEffect(() => {
    let raf = 0;
    const tick = (ts: number) => {
      if (!startRef.current) startRef.current = ts;
      const elapsed = ts - startRef.current - delay * 1000;
      let t = ((elapsed % DURATION) + DURATION) % DURATION;
      t = t / DURATION;
      const path = pathRef.current;
      const dot = dotRef.current;
      if (path && dot) {
        const pt = path.getPointAtLength(length * t);
        dot.setAttribute("cx", String(pt.x));
        dot.setAttribute("cy", String(pt.y));
        const fade =
          t < 0.05 ? t / 0.05 : t > 0.95 ? (1 - t) / 0.05 : 1;
        dot.setAttribute("opacity", String(fade));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [delay, length]);

  return (
    <>
      <path ref={pathRef} d={d} fill="none" stroke="none" />
      <circle
        ref={dotRef}
        r={5}
        fill={color}
        style={{ filter: `drop-shadow(0 0 8px ${color})` }}
      />
    </>
  );
}

// ---------- Tooltip ----------
function Tooltip({
  visible,
  x,
  y,
  children,
}: {
  visible: boolean;
  x: number;
  y: number;
  children: React.ReactNode;
}) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.15 }}
          style={{ left: x + 14, top: y + 14 }}
          className="pointer-events-none fixed z-50 max-w-[300px] rounded-lg bg-foreground px-3 py-2 font-mono text-[11px] leading-snug text-background shadow-2xl"
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ---------- Main component ----------
export default function StoolNetDiagram() {
  const [backbone, setBackbone] = useState<Backbone>("dense");
  const [hovered, setHovered] = useState<BranchKey | null>(null);
  const [mathsOpen, setMathsOpen] = useState(false);
  const [tip, setTip] = useState<{
    show: boolean;
    x: number;
    y: number;
    node: React.ReactNode;
  }>({ show: false, x: 0, y: 0, node: null });

  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLDivElement>(null);
  const backboneRef = useRef<HTMLDivElement>(null);
  const tripleRef = useRef<HTMLDivElement>(null);
  const fusionRef = useRef<HTMLDivElement>(null);
  const gapRef = useRef<HTMLDivElement>(null);
  const branchRefs = useRef<Record<BranchKey, HTMLElement | null>>({
    pos: null,
    chan: null,
    type: null,
  });
  const headRefs = useRef<HTMLElement[]>([]);

  const { paths, size } = useConnectors(
    containerRef,
    {
      input: inputRef,
      backbone: backboneRef,
      triple: tripleRef,
      fusion: fusionRef,
      gap: gapRef,
    },
    branchRefs,
    headRefs,
    [backbone],
  );

  const colorFor = (kind: PathInfo["kind"]) => {
    if (kind === "pos") return "var(--pos)";
    if (kind === "chan") return "var(--chan)";
    if (kind === "type") return "var(--type)";
    return "var(--foreground)";
  };

  const showTip = (e: React.MouseEvent, node: React.ReactNode) => {
    setTip({ show: true, x: e.clientX, y: e.clientY, node });
  };
  const moveTip = (e: React.MouseEvent) => {
    setTip((t) => ({ ...t, x: e.clientX, y: e.clientY }));
  };
  const hideTip = () => setTip((t) => ({ ...t, show: false }));

  return (
    <div className="min-h-screen bg-background px-6 pb-20 pt-8 text-foreground">
      <div className="mx-auto max-w-[1280px]">
        {/* Header */}
        <header className="mb-8 flex flex-wrap items-end justify-between gap-6 border-b border-border pb-6">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-soft">
              Architecture Diagram · Interactive
            </div>
            <h1 className="mt-2 font-display text-4xl font-extrabold leading-[1.05] tracking-tight md:text-5xl">
              StoolNetTriple
              <span className="block text-ink-soft text-2xl font-bold mt-1 md:text-3xl">
                Triple Attention CNN
              </span>
            </h1>
            <p className="mt-3 max-w-xl text-sm text-ink-soft">
              A multi-task deep network for Bristol Stool Scale (BSS)
              classification — joining a pretrained backbone with three parallel
              attention mechanisms feeding three classification heads.
            </p>
          </div>

          <div className="flex flex-col items-end gap-3">
            <div className="inline-flex rounded-full border border-border bg-surface-soft p-1 font-mono text-xs">
              {(["dense", "resnet"] as Backbone[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setBackbone(k)}
                  className="relative rounded-full px-4 py-2 font-medium transition-colors"
                >
                  {backbone === k && (
                    <motion.span
                      layoutId="bb-pill"
                      className="absolute inset-0 rounded-full bg-foreground"
                      transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span
                    className={`relative z-10 ${
                      backbone === k ? "text-background" : "text-ink-soft"
                    }`}
                  >
                    {BACKBONE[k].name} · {BACKBONE[k].dim}-dim
                  </span>
                </button>
              ))}
            </div>
            <button
              onClick={() => setMathsOpen(true)}
              className="rounded-full border border-border bg-surface px-4 py-2 font-mono text-xs text-foreground transition-all hover:border-foreground hover:bg-foreground hover:text-background"
            >
              Show Maths →
            </button>
          </div>
        </header>

        {/* Diagram */}
        <div ref={containerRef} className="relative">
          {/* SVG connectors */}
          <svg
            className="pointer-events-none absolute inset-0 z-0 overflow-visible"
            width={size.w}
            height={size.h}
            viewBox={`0 0 ${size.w} ${size.h}`}
          >
            {paths.map((p, i) => (
              <motion.path
                key={`${p.kind}-${i}-${p.d.length}`}
                d={p.d}
                fill="none"
                stroke="var(--border)"
                strokeWidth={1.4}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{
                  pathLength: { duration: 0.9, delay: p.delay, ease: "easeInOut" },
                  opacity: { duration: 0.3, delay: p.delay },
                }}
              />
            ))}
            {paths.map((p, i) => (
              <FlowPulse
                key={`pulse-${i}-${p.d.length}`}
                d={p.d}
                length={p.length}
                delay={p.delay}
                color={colorFor(p.kind)}
              />
            ))}
          </svg>

          <div className="relative z-10 flex flex-col items-center gap-5 py-6">
            {/* Input */}
            <Node
              ref={inputRef as React.RefObject<HTMLDivElement>}
              variant="input"
              title="Input Image"
              meta="224 × 224 × 3 · ImageNet-norm"
              delay={0}
            />

            {/* Backbone */}
            <Node
              ref={backboneRef as React.RefObject<HTMLDivElement>}
              variant="backbone"
              delay={0.1}
            >
              <AnimatePresence mode="wait">
                <motion.div
                  key={backbone}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.25 }}
                >
                  <div className="font-display text-sm font-bold">
                    {BACKBONE[backbone].name} (Pretrained)
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-ink-soft">
                    Feature map: H × W ×{" "}
                    <span className="font-semibold text-foreground">
                      {BACKBONE[backbone].dim}
                    </span>
                  </div>
                  <div className="mt-1 font-mono text-[10.5px] text-ink-mute">
                    Layers frozen → unfrozen at epoch 20
                  </div>
                </motion.div>
              </AnimatePresence>
            </Node>

            {/* Triple Attention */}
            <motion.div
              ref={tripleRef}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative w-full max-w-[1180px] rounded-3xl border border-dashed border-border bg-gradient-to-b from-surface to-surface-soft p-6 pt-8"
            >
              <div className="absolute -top-3 left-6 bg-background px-2.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-soft">
                Triple Attention Module · 3 parallel branches
              </div>

              <LayoutGroup>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                  {BRANCHES.map((b, i) => {
                    const isHover = hovered === b.key;
                    const dim = hovered !== null && !isHover;
                    return (
                      <motion.div
                        key={b.key}
                        ref={(el) => {
                          branchRefs.current[b.key] = el;
                        }}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{
                          opacity: dim ? 0.32 : 1,
                          y: 0,
                          scale: isHover ? 1.015 : dim ? 0.985 : 1,
                        }}
                        transition={{
                          opacity: { duration: 0.35 },
                          y: { duration: 0.5, delay: 0.25 + i * 0.08 },
                          scale: { type: "spring", stiffness: 280, damping: 24 },
                        }}
                        onMouseEnter={(e) => {
                          setHovered(b.key);
                          showTip(
                            e,
                            <>
                              <strong className="font-display text-white">
                                {b.title}
                              </strong>
                              <span className="mt-1 block text-[#cfcfd6]">
                                {b.tip}
                              </span>
                              <span className="mt-1 block text-white">
                                {b.formula}
                              </span>
                            </>,
                          );
                        }}
                        onMouseMove={moveTip}
                        onMouseLeave={() => {
                          setHovered(null);
                          hideTip();
                        }}
                        className={`group relative cursor-pointer rounded-2xl border bg-card p-4 shadow-sm transition-shadow hover:shadow-2xl branch-${b.key}`}
                        style={{
                          borderColor:
                            b.key === "pos"
                              ? "var(--pos)"
                              : b.key === "chan"
                                ? "var(--chan)"
                                : "var(--type)",
                        }}
                      >
                        <div className="absolute right-3 top-3 rounded-full bg-surface-soft px-2 py-0.5 font-mono text-[10px] text-ink-soft">
                          {b.index}
                        </div>
                        <h3
                          className="font-display text-[15px] font-bold tracking-tight"
                          style={{
                            color:
                              b.key === "pos"
                                ? "var(--pos-deep)"
                                : b.key === "chan"
                                  ? "var(--chan-deep)"
                                  : "var(--type-deep)",
                          }}
                        >
                          {b.title}
                        </h3>
                        <div className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.08em] text-ink-soft">
                          {b.sub}
                        </div>
                        <ol className="flex flex-col gap-1.5">
                          {b.steps.map((s, j) => (
                            <motion.li
                              key={j}
                              initial={{ opacity: 0, x: -6 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{
                                delay: 0.45 + i * 0.08 + j * 0.04,
                                duration: 0.3,
                              }}
                              className="rounded-md bg-surface-soft px-2.5 py-1.5 font-mono text-[11.5px] leading-snug"
                              style={{
                                borderLeft: `3px solid ${
                                  b.key === "pos"
                                    ? "var(--pos)"
                                    : b.key === "chan"
                                      ? "var(--chan)"
                                      : "var(--type)"
                                }`,
                              }}
                            >
                              <span className="text-ink-mute">{j + 1} ·</span>{" "}
                              {s}
                            </motion.li>
                          ))}
                        </ol>
                        <div
                          className="mt-3 rounded-md border border-dashed px-2.5 py-2 font-mono text-[11px]"
                          style={{
                            borderColor:
                              b.key === "pos"
                                ? "var(--pos)"
                                : b.key === "chan"
                                  ? "var(--chan)"
                                  : "var(--type)",
                            background:
                              b.key === "pos"
                                ? "var(--pos-soft)"
                                : b.key === "chan"
                                  ? "var(--chan-soft)"
                                  : "var(--type-soft)",
                            color:
                              b.key === "pos"
                                ? "var(--pos-deep)"
                                : b.key === "chan"
                                  ? "var(--chan-deep)"
                                  : "var(--type-deep)",
                          }}
                        >
                          {b.formula}
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </LayoutGroup>
            </motion.div>

            {/* Fusion */}
            <Node
              ref={fusionRef as React.RefObject<HTMLDivElement>}
              variant="fusion"
              title="Fusion · Equal-weight Average"
              meta="(Position + Channel + Type) / 3"
              delay={0.4}
            />

            {/* GAP */}
            <Node
              ref={gapRef as React.RefObject<HTMLDivElement>}
              variant="gap"
              title="GAP → Dropout → Flatten"
              delay={0.5}
            >
              <div className="mt-1 font-mono text-[11px] text-ink-soft">
                Shared feature vector ·{" "}
                <AnimatePresence mode="wait">
                  <motion.span
                    key={backbone}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.2 }}
                    className="inline-block font-semibold text-foreground"
                  >
                    {BACKBONE[backbone].dim}
                  </motion.span>
                </AnimatePresence>
                -d
              </div>
            </Node>

            {/* Heads */}
            <div className="grid w-full max-w-[1000px] grid-cols-1 gap-4 md:grid-cols-3">
              {HEADS.map((h, i) => (
                <motion.div
                  key={h.key}
                  ref={(el) => {
                    if (el) headRefs.current[i] = el;
                  }}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 + i * 0.08 }}
                  whileHover={{ y: -4 }}
                  onMouseEnter={(e) =>
                    showTip(
                      e,
                      <>
                        <strong className="font-display text-white">
                          {h.title}
                        </strong>
                        <span className="mt-1 block">Arch: {h.arch}</span>
                        <span className="block">Loss: {h.loss}</span>
                        <span className="block">Classes: {h.classes}</span>
                      </>,
                    )
                  }
                  onMouseMove={moveTip}
                  onMouseLeave={hideTip}
                  className="relative cursor-pointer overflow-hidden rounded-xl border border-border bg-card p-4 shadow-sm transition-shadow hover:shadow-2xl"
                >
                  <div
                    className="absolute left-4 right-4 top-0 h-[3px] rounded-b"
                    style={{ background: h.color }}
                  />
                  <h4 className="mt-1 font-display text-sm font-bold">
                    {h.title}
                  </h4>
                  <div className="mb-1.5 font-mono text-xs text-ink-soft">
                    {h.arch}
                  </div>
                  <span className="inline-block rounded-full bg-surface-soft px-2 py-0.5 font-mono text-[11px]">
                    {h.lossShort}
                  </span>
                  <div className="mt-2 font-mono text-[10.5px] text-ink-soft">
                    {h.classes}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Legend */}
        <footer className="mt-10 flex flex-wrap items-center gap-4 border-t border-border pt-5 font-mono text-[11px] text-ink-soft">
          <Swatch c="var(--pos)" l="Position Attention" />
          <Swatch c="var(--chan)" l="Channel Attention" />
          <Swatch c="var(--type)" l="Type Attention" />
          <Swatch c="var(--fusion)" l="Fusion" />
          <Swatch c="var(--head-type)" l="Type Head" />
          <Swatch c="var(--head-shape)" l="Shape Head" />
          <Swatch c="var(--head-color)" l="Color Head" />
          <span className="ml-auto">
            L = 2.0·L_type + 0.8·L_shape + 0.1·L_color
          </span>
        </footer>
      </div>

      {/* Maths panel */}
      <AnimatePresence>
        {mathsOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMathsOpen(false)}
              className="fixed inset-0 z-40 bg-foreground/20 backdrop-blur-[2px]"
            />
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", stiffness: 320, damping: 36 }}
              className="fixed right-0 top-0 z-50 h-full w-[min(420px,92vw)] overflow-y-auto border-l border-border bg-card p-7 shadow-2xl"
            >
              <button
                onClick={() => setMathsOpen(false)}
                className="absolute right-4 top-4 text-2xl text-ink-soft hover:text-foreground"
              >
                ×
              </button>
              <h2 className="font-display text-2xl font-extrabold">
                Key Equations
              </h2>
              <div className="mb-5 mt-1 font-mono text-[11px] uppercase tracking-[0.08em] text-ink-soft">
                Triple Attention · Multi-task Loss
              </div>
              <Eq title="Position Attention" color="var(--pos)">
                Attention(Q,K,V) = Softmax(QKᵀ / √d) · V
              </Eq>
              <Eq title="Channel Attention" color="var(--chan)">
                A_c = Softmax(C · Cᵀ) ; X′ = A_c · X
              </Eq>
              <Eq title="Type Attention" color="var(--type)">
                MS = Σ_k SE(Conv_k(x)), k ∈ {`{3,5,7}`}
              </Eq>
              <Eq title="Residual (all branches)" color="var(--foreground)">
                out = γ · attention(x) + x, γ₀ = 0.1
              </Eq>
              <Eq title="Fusion" color="var(--fusion)">
                F = (P + C + T) / 3
              </Eq>
              <Eq title="Combined Loss" color="var(--head-type)">
                L = 2.0·L_type + 0.8·L_shape + 0.1·L_color
              </Eq>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <Tooltip visible={tip.show} x={tip.x} y={tip.y}>
        {tip.node}
      </Tooltip>
    </div>
  );
}

// ---------- Subcomponents ----------
const Node = ({
  ref,
  variant,
  title,
  meta,
  children,
  delay = 0,
}: {
  ref: React.RefObject<HTMLDivElement>;
  variant: "input" | "backbone" | "fusion" | "gap";
  title?: string;
  meta?: string;
  children?: React.ReactNode;
  delay?: number;
}) => {
  const styles: Record<string, string> = {
    input: "bg-card border-border min-w-[200px]",
    backbone:
      "bg-gradient-to-b from-card to-surface-soft border-foreground/15 min-w-[280px]",
    fusion: "border-fusion min-w-[260px]",
    gap: "bg-card border-border min-w-[220px]",
  };
  const fusionStyle =
    variant === "fusion" ? { background: "var(--fusion-soft)" } : {};

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={`relative z-10 rounded-2xl border px-5 py-3.5 text-center font-mono text-xs shadow-[0_8px_24px_-16px_rgba(0,0,0,0.18)] ${styles[variant]}`}
      style={fusionStyle}
    >
      {title && (
        <div className="font-display text-[13px] font-bold tracking-tight">
          {title}
        </div>
      )}
      {meta && (
        <div className="mt-0.5 font-mono text-[11px] text-ink-soft">{meta}</div>
      )}
      {children}
    </motion.div>
  );
};

const Swatch = ({ c, l }: { c: string; l: string }) => (
  <span className="inline-flex items-center gap-1.5">
    <span
      className="inline-block h-2.5 w-2.5 rounded-sm"
      style={{ background: c }}
    />
    {l}
  </span>
);

const Eq = ({
  title,
  color,
  children,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
}) => (
  <motion.div
    initial={{ opacity: 0, x: 10 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.3 }}
    className="mb-4"
  >
    <h4 className="mb-1.5 font-display text-[13px] font-bold">{title}</h4>
    <div
      className="overflow-x-auto rounded-md bg-surface-soft px-3 py-2.5 font-mono text-[12.5px]"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      {children}
    </div>
  </motion.div>
);
