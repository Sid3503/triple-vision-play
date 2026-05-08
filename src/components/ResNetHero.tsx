import { motion } from "framer-motion";

// ─── 5-Fold CV animated visualization ────────────────────────────────────────

const FOLDS_DATA = [
  { acc: 76.9, color: "#1D9E75", label: "Best" },
  { acc: 53.8, color: "#C07A0A", label: "" },
  { acc: 61.5, color: "#C07A0A", label: "" },
  { acc: 58.3, color: "#C07A0A", label: "" },
  { acc: 50.0, color: "#B83820", label: "Weakest" },
];

function FoldVisSVG() {
  const TOTAL_W = 200;
  const TRAIN_FRAC = 0.79;

  return (
    <svg viewBox="0 0 380 210" className="w-full" aria-hidden="true">
      {/* Column headers */}
      <text x="6"  y="12" fontSize="8" fill="var(--ink-mute)" fontFamily="monospace">Fold</text>
      <text x="34" y="12" fontSize="8" fill="#1D9E75" fontFamily="monospace" fontWeight="600">train</text>
      <text x={34 + TOTAL_W * TRAIN_FRAC + 4} y="12" fontSize="8" fill="#C07A0A" fontFamily="monospace" fontWeight="600">val</text>
      <text x={34 + TOTAL_W + 12} y="12" fontSize="8" fill="var(--ink-mute)" fontFamily="monospace">acc</text>
      <text x={34 + TOTAL_W + 56} y="12" fontSize="8" fill="var(--ink-mute)" fontFamily="monospace">model</text>

      {FOLDS_DATA.map((f, i) => {
        const y = 22 + i * 34;
        const barH = 18;
        const trainW = TOTAL_W * TRAIN_FRAC;
        const valW = TOTAL_W * (1 - TRAIN_FRAC);
        const accBarW = ((f.acc - 45) / 38) * 60;

        return (
          <g key={i}>
            {/* Fold label */}
            <text x="6" y={y + barH / 2 + 4} fontSize="8" fill="var(--ink-mute)"
              fontFamily="monospace" fontWeight={f.label ? "700" : "400"}>F{i + 1}</text>

            {/* Train segment */}
            <motion.rect
              x="34" y={y} width={trainW} height={barH} rx="3"
              fill="#1D9E75" opacity={0.22}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              style={{ transformOrigin: "34px center" }}
              transition={{ duration: 0.9, delay: 0.3 + i * 0.15, ease: [0.22, 1, 0.36, 1] }}
            />

            {/* Val segment (highlighted differently per fold) */}
            <motion.rect
              x={34 + trainW} y={y} width={valW} height={barH} rx="3"
              fill="#C07A0A" opacity={0.38}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              style={{ transformOrigin: `${34 + trainW}px center` }}
              transition={{ duration: 0.6, delay: 0.5 + i * 0.15, ease: [0.22, 1, 0.36, 1] }}
            />

            {/* Val fold indicator */}
            <motion.rect
              x={34 + trainW + i * (valW / 5)} y={y} width={valW / 5} height={barH} rx="2"
              fill="#C07A0A" opacity={0.75}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              style={{ transformOrigin: `${34 + trainW + i * (valW / 5)}px center` }}
              transition={{ duration: 0.5, delay: 0.65 + i * 0.15, ease: [0.22, 1, 0.36, 1] }}
            />

            {/* Accuracy bar */}
            <rect x={34 + TOTAL_W + 12} y={y + 4} width={60} height={barH - 8} rx="2"
              fill="var(--surface-soft)" />
            <motion.rect
              x={34 + TOTAL_W + 12} y={y + 4} width={accBarW} height={barH - 8} rx="2"
              fill={f.color}
              initial={{ width: 0 }}
              animate={{ width: accBarW }}
              transition={{ duration: 1.1, delay: 0.8 + i * 0.15, ease: [0.22, 1, 0.36, 1] }}
            />
            <text x={34 + TOTAL_W + 76} y={y + barH / 2 + 3} fontSize="7.5" fill={f.color}
              fontFamily="monospace" fontWeight="700">{f.acc}%</text>

            {/* ResNet model box */}
            <motion.rect
              x={34 + TOTAL_W + 114} y={y + 1} width={46} height={barH - 2} rx="3"
              fill="#EEF4FF" stroke="#5B8FD0" strokeWidth="1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 1 + i * 0.15 }}
            />
            <text x={34 + TOTAL_W + 137} y={y + barH / 2 + 3} textAnchor="middle" fontSize="6.5"
              fill="#3B68B0" fontFamily="monospace">ResNet50</text>

            {/* Badge */}
            {f.label && (
              <text x={34 + TOTAL_W + 164} y={y + barH / 2 + 3} fontSize="7"
                fill={f.color} fontFamily="monospace" fontWeight="600">{f.label}</text>
            )}
          </g>
        );
      })}

      {/* Ensemble line */}
      <line x1="34" y1="196" x2={34 + TOTAL_W + 165} y2="196"
        stroke="var(--border)" strokeWidth="1" strokeDasharray="3 2" />
      <text x="34" y="207" fontSize="7.5" fill="var(--ink-mute)" fontFamily="monospace">
        Ensemble mean: 60.1% ± 9.3%  ·  ×8 TTA  ·  40 forward passes / image
      </text>
    </svg>
  );
}

// ─── Animated blobs ───────────────────────────────────────────────────────────

function Blob({
  color, size, top, left, xAnim, yAnim, duration, delay,
}: {
  color: string; size: number; top: string; left: string;
  xAnim: number[]; yAnim: number[]; duration: number; delay: number;
}) {
  return (
    <motion.div
      className="pointer-events-none absolute rounded-full"
      style={{
        width: size, height: size, top, left,
        background: color, opacity: 0.14, filter: "blur(90px)",
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
      <span className="font-mono text-[10px] uppercase tracking-[.14em]"
        style={{ color: "var(--ink-mute)" }}>Scroll to explore</span>
      <motion.div
        animate={{ y: [0, 7, 0] }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
      >
        <svg width="16" height="10" viewBox="0 0 16 10" fill="none">
          <path d="M1 1L8 8L15 1" stroke="var(--ink-mute)" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </motion.div>
    </motion.div>
  );
}

// ─── Stagger variants ─────────────────────────────────────────────────────────

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
};
const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

// ─── Main hero ────────────────────────────────────────────────────────────────

export function ResNetHero() {
  return (
    <section className="relative min-h-[92vh] overflow-hidden flex flex-col items-center justify-center">
      {/* Blobs */}
      <Blob color="#1D9E75" size={600} top="-8%" left="52%"
        xAnim={[0, 35, -25, 20, 0]} yAnim={[0, 25, -15, 10, 0]}
        duration={13} delay={0} />
      <Blob color="#5B8FD0" size={500} top="38%" left="-6%"
        xAnim={[0, -25, 20, -10, 0]} yAnim={[0, -20, 25, -8, 0]}
        duration={10} delay={2} />
      <Blob color="#C07A0A" size={440} top="18%" left="28%"
        xAnim={[0, 20, -12, 0]} yAnim={[0, -15, 22, 0]}
        duration={8.5} delay={3.5} />

      {/* Dot grid */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(circle, var(--border) 1px, transparent 1px)",
          backgroundSize: "28px 28px",
          opacity: 0.5,
        }}
      />

      {/* Bottom fade */}
      <div
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-40"
        style={{ background: "linear-gradient(to bottom, transparent, var(--background))" }}
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
                style={{ background: "var(--surface-soft)", border: "1px solid var(--border)", color: "var(--ink-soft)" }}
              >
                Transfer Learning
              </span>
              <span
                className="rounded-full px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-[.12em]"
                style={{ background: "#E8F5EF", border: "1px solid #1D9E75", color: "#1D7A5C" }}
              >
                5-Fold Cross Validation
              </span>
            </motion.div>

            {/* Title */}
            <motion.h1
              variants={item}
              className="font-display font-extrabold leading-[1.02] tracking-tight"
              style={{ fontSize: "clamp(2.6rem, 6.5vw, 4.8rem)" }}
            >
              ResNet
              <span style={{ color: "#3B68B0" }}>50</span>
              <span
                className="block font-display font-bold leading-tight"
                style={{ fontSize: "clamp(1.2rem, 3vw, 2rem)", color: "var(--ink-soft)" }}
              >
                5-Fold CV Training Pipeline
              </span>
            </motion.h1>

            {/* Description */}
            <motion.p
              variants={item}
              className="mt-4 max-w-lg text-sm leading-relaxed"
              style={{ color: "var(--ink-soft)" }}
            >
              Adapts a pretrained ResNet-50 backbone to a small 63-image clinical dataset using
              stratified 5-fold cross-validation. All five fold models are ensembled at inference
              with 8-augmentation TTA for robust Bristol Stool Scale prediction.
            </motion.p>

            {/* Strategy pills */}
            <motion.div variants={item} className="mt-6 flex flex-wrap gap-2">
              {[
                { label: "MixUp α=0.3",    bg: "#FFF3E0", bd: "#FFCC80", color: "#EF9F27" },
                { label: "CutMix p=0.5",   bg: "#FDF0EA", bd: "#F0B090", color: "#D85A30" },
                { label: "EMA 0.999",      bg: "#EBF3FD", bd: "#A8CCF2", color: "#378ADD" },
                { label: "×8 TTA",         bg: "#E8F5EF", bd: "#88D4B0", color: "#1D7A5C" },
                { label: "Focal Loss γ=2", bg: "var(--surface-soft)", bd: "var(--border)", color: "var(--ink-soft)" },
              ].map(({ label, bg, bd, color }) => (
                <span
                  key={label}
                  className="rounded-lg px-3 py-1.5 font-mono text-[10px] font-semibold"
                  style={{ background: bg, border: `1px solid ${bd}`, color }}
                >
                  {label}
                </span>
              ))}
            </motion.div>

            {/* Stats */}
            <motion.div variants={item} className="mt-8 flex gap-8">
              {[
                { val: "5", label: "Folds", color: "#3B68B0" },
                { val: "63", label: "Images total", color: "#1D9E75" },
                { val: "60.1%", label: "Ensemble mean acc", color: "#C07A0A" },
              ].map(({ val, label, color }) => (
                <div key={label}>
                  <div className="font-display font-extrabold leading-none"
                    style={{ fontSize: "clamp(1.6rem, 4vw, 2.5rem)", color }}>
                    {val}
                  </div>
                  <div className="mt-1 font-mono text-[10px] uppercase tracking-[.08em]"
                    style={{ color: "var(--ink-mute)" }}>
                    {label}
                  </div>
                </div>
              ))}
            </motion.div>
          </motion.div>

          {/* ── Right: 5-fold visualization ── */}
          <motion.div
            className="hidden lg:block shrink-0 w-[420px]"
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
              <div className="mb-3 font-mono text-[9px] uppercase tracking-[.12em]"
                style={{ color: "var(--ink-mute)" }}>
                Training splits · live render
              </div>
              <FoldVisSVG />
            </div>
          </motion.div>
        </div>
      </div>

      <ScrollIndicator />
    </section>
  );
}
