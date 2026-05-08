import { motion } from "framer-motion";

// ─── Neural network node layout (matches StoolNetTriple architecture) ─────────

const NODES: { id: string; x: number; y: number; r: number; color: string; layer: number }[] = [
  // Input
  { id: "i0", x: 28, y: 72, r: 5, color: "var(--foreground)", layer: 0 },
  { id: "i1", x: 28, y: 128, r: 5, color: "var(--foreground)", layer: 0 },
  { id: "i2", x: 28, y: 184, r: 5, color: "var(--foreground)", layer: 0 },
  // Backbone
  { id: "b0", x: 104, y: 55, r: 5.5, color: "var(--ink-soft)", layer: 1 },
  { id: "b1", x: 104, y: 105, r: 5.5, color: "var(--ink-soft)", layer: 1 },
  { id: "b2", x: 104, y: 155, r: 5.5, color: "var(--ink-soft)", layer: 1 },
  { id: "b3", x: 104, y: 205, r: 5.5, color: "var(--ink-soft)", layer: 1 },
  // Position attention
  { id: "p0", x: 190, y: 42, r: 6, color: "var(--pos)", layer: 2 },
  { id: "p1", x: 190, y: 90, r: 6, color: "var(--pos)", layer: 2 },
  // Channel attention
  { id: "c0", x: 190, y: 138, r: 6, color: "var(--chan)", layer: 3 },
  { id: "c1", x: 190, y: 186, r: 6, color: "var(--chan)", layer: 3 },
  // Type attention
  { id: "t0", x: 190, y: 234, r: 6, color: "var(--type)", layer: 4 },
  // Fusion
  { id: "f0", x: 272, y: 88, r: 7, color: "var(--fusion)", layer: 5 },
  { id: "f1", x: 272, y: 184, r: 7, color: "var(--fusion)", layer: 5 },
  // Heads
  { id: "h0", x: 356, y: 66, r: 6.5, color: "var(--head-type)", layer: 6 },
  { id: "h1", x: 356, y: 138, r: 6.5, color: "var(--head-shape)", layer: 6 },
  { id: "h2", x: 356, y: 210, r: 6.5, color: "var(--head-color)", layer: 6 },
];

const EDGES: [string, string, number][] = [
  // input → backbone
  ["i0", "b0", 0],
  ["i0", "b1", 0],
  ["i1", "b0", 0],
  ["i1", "b1", 0],
  ["i1", "b2", 0],
  ["i2", "b2", 0],
  ["i2", "b3", 0],
  // backbone → attention branches
  ["b0", "p0", 1],
  ["b1", "p0", 1],
  ["b1", "p1", 1],
  ["b2", "p1", 1],
  ["b1", "c0", 1],
  ["b2", "c0", 1],
  ["b2", "c1", 1],
  ["b3", "c1", 1],
  ["b3", "t0", 1],
  ["b2", "t0", 1],
  // attention → fusion
  ["p0", "f0", 2],
  ["p1", "f0", 2],
  ["c0", "f0", 2],
  ["c1", "f1", 2],
  ["t0", "f1", 2],
  // fusion → heads
  ["f0", "h0", 3],
  ["f0", "h1", 3],
  ["f1", "h1", 3],
  ["f1", "h2", 3],
];

const LAYER_DELAY = 0.45;

function NeuralNetSVG() {
  const nodeMap = Object.fromEntries(NODES.map((n) => [n.id, n]));

  return (
    <svg viewBox="0 0 400 280" className="w-full max-w-[420px]" aria-hidden="true">
      {/* Edges */}
      {EDGES.map(([a, b, waveLayer]) => {
        const na = nodeMap[a];
        const nb = nodeMap[b];
        const delay = waveLayer * LAYER_DELAY;
        return (
          <motion.line
            key={`${a}-${b}`}
            x1={na.x}
            y1={na.y}
            x2={nb.x}
            y2={nb.y}
            stroke={nb.color}
            strokeWidth="1"
            initial={{ opacity: 0.08 }}
            animate={{ opacity: [0.08, 0.38, 0.08] }}
            transition={{
              duration: 2.8,
              delay,
              repeat: Infinity,
              ease: "easeInOut",
              repeatDelay: 1.2,
            }}
          />
        );
      })}

      {/* Nodes */}
      {NODES.map((n) => {
        const delay = n.layer * LAYER_DELAY;
        return (
          <g key={n.id}>
            {/* Glow ring */}
            <motion.circle
              cx={n.x}
              cy={n.y}
              r={n.r + 4}
              fill={n.color}
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.18, 0] }}
              transition={{
                duration: 2.8,
                delay,
                repeat: Infinity,
                ease: "easeInOut",
                repeatDelay: 1.2,
              }}
            />
            {/* Core */}
            <motion.circle
              cx={n.x}
              cy={n.y}
              r={n.r}
              fill={n.color}
              initial={{ opacity: 0.3, scale: 1 }}
              animate={{ opacity: [0.3, 0.9, 0.3], scale: [1, 1.25, 1] }}
              transition={{
                duration: 2.8,
                delay,
                repeat: Infinity,
                ease: "easeInOut",
                repeatDelay: 1.2,
              }}
            />
          </g>
        );
      })}

      {/* Layer labels */}
      {[
        { x: 28, label: "Input" },
        { x: 104, label: "Backbone" },
        { x: 190, label: "Attention" },
        { x: 272, label: "Fusion" },
        { x: 356, label: "Heads" },
      ].map(({ x, label }) => (
        <text
          key={label}
          x={x}
          y={268}
          textAnchor="middle"
          fontSize="8"
          fill="var(--ink-mute)"
          fontFamily="monospace"
        >
          {label}
        </text>
      ))}

      {/* Attention type labels */}
      <text x="222" y="68" fontSize="7" fill="var(--pos)" fontFamily="monospace" fontWeight="600">
        Pos
      </text>
      <text x="222" y="162" fontSize="7" fill="var(--chan)" fontFamily="monospace" fontWeight="600">
        Chan
      </text>
      <text x="222" y="238" fontSize="7" fill="var(--type)" fontFamily="monospace" fontWeight="600">
        Type
      </text>
    </svg>
  );
}

// ─── Animated background blobs ────────────────────────────────────────────────

function Blob({
  color,
  size,
  top,
  left,
  xAnim,
  yAnim,
  duration,
  delay,
}: {
  color: string;
  size: number;
  top: string;
  left: string;
  xAnim: number[];
  yAnim: number[];
  duration: number;
  delay: number;
}) {
  return (
    <motion.div
      className="pointer-events-none absolute rounded-full"
      style={{
        width: size,
        height: size,
        top,
        left,
        background: color,
        opacity: 0.16,
        filter: "blur(90px)",
      }}
      animate={{ x: xAnim, y: yAnim }}
      transition={{ duration, repeat: Infinity, ease: "easeInOut", repeatType: "mirror", delay }}
    />
  );
}

// ─── Scroll indicator ─────────────────────────────────────────────────────────

function ScrollIndicator() {
  return (
    <motion.div
      className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 2.2, duration: 0.8 }}
    >
      <span
        className="font-mono text-[10px] uppercase tracking-[.14em]"
        style={{ color: "var(--ink-mute)" }}
      >
        Scroll to explore
      </span>
      <motion.div
        animate={{ y: [0, 7, 0] }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
      >
        <svg width="16" height="10" viewBox="0 0 16 10" fill="none">
          <path
            d="M1 1L8 8L15 1"
            stroke="var(--ink-mute)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </motion.div>
    </motion.div>
  );
}

// ─── Stagger container ────────────────────────────────────────────────────────

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.13 } },
};
const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

// ─── Main hero component ──────────────────────────────────────────────────────

export function StoolNetHero() {
  return (
    <section className="relative min-h-[92vh] overflow-hidden flex flex-col items-center justify-center">
      {/* Atmospheric blobs */}
      <Blob
        color="var(--pos)"
        size={640}
        top="-12%"
        left="55%"
        xAnim={[0, 40, -20, 30, 0]}
        yAnim={[0, 30, -20, 10, 0]}
        duration={14}
        delay={0}
      />
      <Blob
        color="var(--chan)"
        size={540}
        top="40%"
        left="-8%"
        xAnim={[0, -30, 20, -15, 0]}
        yAnim={[0, -30, 25, -10, 0]}
        duration={11}
        delay={2}
      />
      <Blob
        color="var(--type)"
        size={480}
        top="15%"
        left="20%"
        xAnim={[0, 25, -15, 0]}
        yAnim={[0, -20, 30, 0]}
        duration={9}
        delay={4}
      />

      {/* Dot grid overlay */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(circle, var(--border) 1px, transparent 1px)",
          backgroundSize: "28px 28px",
          opacity: 0.5,
        }}
      />

      {/* Vignette fade to background at bottom */}
      <div
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-40"
        style={{
          background: "linear-gradient(to bottom, transparent, var(--background))",
        }}
      />

      {/* Content */}
      <div className="relative z-10 mx-auto w-full max-w-[1060px] px-5 pb-24 pt-12">
        <div className="flex items-center gap-12 lg:gap-20">
          {/* ── Left: text ── */}
          <motion.div
            className="flex-1 min-w-0"
            variants={container}
            initial="hidden"
            animate="show"
          >
            {/* Eyebrow */}
            <motion.div variants={item} className="mb-5 flex items-center gap-3">
              <span
                className="rounded-full px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-[.12em]"
                style={{
                  background: "var(--surface-soft)",
                  border: "1px solid var(--border)",
                  color: "var(--ink-soft)",
                }}
              >
                Research · Deep Learning
              </span>
              <span
                className="rounded-full px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-[.12em]"
                style={{
                  background: "var(--pos-soft)",
                  border: "1px solid var(--pos)",
                  color: "var(--pos-deep)",
                }}
              >
                Bristol Stool Scale
              </span>
            </motion.div>

            {/* Title */}
            <motion.h1
              variants={item}
              className="font-display font-extrabold leading-[1.02] tracking-tight"
              style={{ fontSize: "clamp(2.8rem, 7vw, 5.2rem)" }}
            >
              Stool
              <span style={{ color: "var(--pos)" }}>Net</span>
              <span style={{ color: "var(--chan)" }}>Triple</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              variants={item}
              className="font-display mt-2 font-semibold"
              style={{ fontSize: "clamp(1rem, 2.5vw, 1.4rem)", color: "var(--ink-soft)" }}
            >
              Triple Attention CNN for Clinical Image Classification
            </motion.p>

            {/* Description */}
            <motion.p
              variants={item}
              className="mt-4 max-w-lg text-sm leading-relaxed"
              style={{ color: "var(--ink-soft)" }}
            >
              A multi-task deep network that joins a pretrained backbone with three specialised
              attention modules — spatial, channel, and texture-ordinal — feeding three parallel
              classification heads for Bristol Stool Scale type, shape, and colour.
            </motion.p>

            {/* Attention branch tags */}
            <motion.div variants={item} className="mt-6 flex flex-wrap gap-2.5">
              {[
                {
                  label: "Position Attention",
                  color: "var(--pos)",
                  bg: "var(--pos-soft)",
                  border: "var(--pos)",
                  sub: "Softmax(QKᵀ/√d)V",
                },
                {
                  label: "Channel Attention",
                  color: "var(--chan-deep)",
                  bg: "var(--chan-soft)",
                  border: "var(--chan)",
                  sub: "Softmax(CCᵀ)·x",
                },
                {
                  label: "Type Attention",
                  color: "var(--type-deep)",
                  bg: "var(--type-soft)",
                  border: "var(--type-deep)",
                  sub: "MS-Conv + SE gate",
                },
              ].map(({ label, color, bg, border, sub }) => (
                <div
                  key={label}
                  className="rounded-xl px-3 py-2"
                  style={{ background: bg, border: `1px solid ${border}` }}
                >
                  <div className="font-display text-[11px] font-bold" style={{ color }}>
                    {label}
                  </div>
                  <div className="font-mono text-[9px]" style={{ color: "var(--ink-mute)" }}>
                    {sub}
                  </div>
                </div>
              ))}
            </motion.div>

            {/* Stats */}
            <motion.div variants={item} className="mt-8 flex gap-8">
              {[
                { val: "3", label: "Attention branches", color: "var(--pos)" },
                { val: "3", label: "Task heads", color: "var(--chan)" },
                { val: "2", label: "Backbone options", color: "var(--type-deep)" },
              ].map(({ val, label, color }) => (
                <div key={label}>
                  <div
                    className="font-display text-4xl font-extrabold leading-none"
                    style={{ color }}
                  >
                    {val}
                  </div>
                  <div
                    className="mt-1 font-mono text-[10px] uppercase tracking-[.08em]"
                    style={{ color: "var(--ink-mute)" }}
                  >
                    {label}
                  </div>
                </div>
              ))}
            </motion.div>
          </motion.div>

          {/* ── Right: animated neural network ── */}
          <motion.div
            className="hidden lg:block shrink-0 w-[400px]"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.9, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
          >
            <div
              className="rounded-3xl p-6"
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                boxShadow: "0 24px 64px -24px rgba(0,0,0,0.10)",
              }}
            >
              <div
                className="mb-3 font-mono text-[9px] uppercase tracking-[.12em]"
                style={{ color: "var(--ink-mute)" }}
              >
                Forward pass · live animation
              </div>
              <NeuralNetSVG />
              <div className="mt-3 flex flex-wrap gap-2">
                {[
                  { label: "DenseNet121", color: "var(--pos)", bg: "var(--pos-soft)" },
                  { label: "ResNet50", color: "var(--ink-soft)", bg: "var(--surface-soft)" },
                ].map(({ label, color, bg }) => (
                  <span
                    key={label}
                    className="rounded-md px-2 py-0.5 font-mono text-[9px]"
                    style={{ background: bg, color, border: "1px solid var(--border)" }}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      <ScrollIndicator />
    </section>
  );
}
