// Architecture overview: Standard DenseNet → Our Triple Attention DenseNet

// ─── SVG helpers ────────────────────────────────────────────────────────────

function FMStack({
  x, y, w = 18, h = 44, color, label, sub, layers = 3,
}: {
  x: number; y: number; w?: number; h?: number;
  color: string; label?: string; sub?: string; layers?: number;
}) {
  const offsets = Array.from({ length: layers }, (_, i) => i);
  return (
    <>
      {offsets.reverse().map((i) => (
        <rect key={i} x={x + i * 3} y={y - i * 3} width={w} height={h}
          fill={color} rx="2.5"
          opacity={i === 0 ? 1 : i === 1 ? 0.55 : 0.3}
        />
      ))}
      {label && (
        <text x={x + w / 2 + (layers - 1) * 1.5} y={y + h + 11} textAnchor="middle"
          fontSize="7.5" fill="var(--ink-soft)" fontFamily="monospace">
          {label}
        </text>
      )}
      {sub && (
        <text x={x + w / 2 + (layers - 1) * 1.5} y={y + h + 21} textAnchor="middle"
          fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace" opacity={0.75}>
          {sub}
        </text>
      )}
    </>
  );
}

function HBox({
  x, y, w = 17, h = 13, label = "H",
}: {
  x: number; y: number; w?: number; h?: number; label?: string;
}) {
  return (
    <>
      <rect x={x} y={y} width={w} height={h} fill="var(--surface-soft)"
        stroke="var(--border)" strokeWidth="1" rx="2" />
      <text x={x + w / 2} y={y + h / 2 + 3} textAnchor="middle"
        fontSize="6.5" fill="var(--ink-soft)" fontFamily="monospace">{label}</text>
    </>
  );
}

function Arrow({ x1, y, x2 }: { x1: number; y: number; x2: number }) {
  return (
    <line x1={x1} y1={y} x2={x2} y2={y} stroke="var(--ink-mute)"
      strokeWidth="1.2" markerEnd="url(#arrowhead)" />
  );
}

function DenseArc({
  x1, y1, x2, y2, peak, color, opacity = 0.55,
}: {
  x1: number; y1: number; x2: number; y2: number;
  peak: number; color: string; opacity?: number;
}) {
  const mx = (x1 + x2) / 2;
  return (
    <path d={`M ${x1} ${y1} C ${mx} ${peak}, ${mx} ${peak}, ${x2} ${y2}`}
      fill="none" stroke={color} strokeWidth="1.2" opacity={opacity}
      strokeDasharray="3 2" />
  );
}

// ─── Standard DenseNet SVG ───────────────────────────────────────────────────

function StandardDenseNetSVG() {
  const stacks = [
    { x: 54,  y: 58, w: 16, h: 44, color: "#D55068", label: "x₀", sub: "3ch" },
    { x: 105, y: 54, w: 20, h: 44, color: "#2A9E8A", label: "x₁" },
    { x: 163, y: 50, w: 24, h: 44, color: "#7A5CC0", label: "x₂" },
    { x: 226, y: 46, w: 28, h: 44, color: "#BF8C20", label: "x₃" },
  ];

  const hBoxes = [
    { x: 76,  y: 72 },
    { x: 131, y: 68 },
    { x: 196, y: 64 },
  ];

  return (
    <svg viewBox="0 0 470 148" className="w-full" role="img" aria-label="Standard DenseNet architecture">
      <defs>
        <marker id="arrowhead-std" markerWidth="6" markerHeight="5"
          refX="5" refY="2.5" orient="auto">
          <polygon points="0 0, 6 2.5, 0 5" fill="var(--ink-mute)" />
        </marker>
      </defs>

      {/* Input image placeholder */}
      <rect x="6" y="64" width="28" height="28" rx="4"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="20" y="81" textAnchor="middle" fontSize="8" fill="var(--ink-mute)" fontFamily="monospace">img</text>

      {/* Arrow input → block */}
      <line x1="34" y1="78" x2="48" y2="78" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-std)" />

      {/* Dense Block 1 dashed border */}
      <rect x="48" y="18" width="222" height="116" rx="6"
        fill="none" stroke="var(--border)" strokeWidth="1" strokeDasharray="4 3" />
      <text x="159" y="13" textAnchor="middle" fontSize="8" fill="var(--ink-mute)"
        fontFamily="monospace">Dense Block</text>

      {/* Dense connection arcs (above stacks) */}
      {/* x₀ → H₂ */}
      <DenseArc x1={66} y1={58} x2={136} y2={68} peak={28} color="#D55068" />
      {/* x₀ → H₃ */}
      <DenseArc x1={66} y1={58} x2={201} y2={64} peak={13} color="#D55068" />
      {/* x₁ → H₃ */}
      <DenseArc x1={118} y1={54} x2={201} y2={64} peak={28} color="#2A9E8A" />

      {/* Feature stacks */}
      {stacks.map((s) => (
        <FMStack key={s.label} {...s} />
      ))}

      {/* H boxes with labels */}
      {hBoxes.map((h, i) => (
        <HBox key={i} {...h} label={`H${i + 1}`} />
      ))}

      {/* Sequential arrows between stacks */}
      <line x1="70" y1="80" x2="76" y2="80" stroke="var(--ink-mute)" strokeWidth="1"
        markerEnd="url(#arrowhead-std)" />
      <line x1="93" y1="76" x2="105" y2="76" stroke="var(--ink-mute)" strokeWidth="1"
        markerEnd="url(#arrowhead-std)" />
      <line x1="125" y1="72" x2="131" y2="72" stroke="var(--ink-mute)" strokeWidth="1"
        markerEnd="url(#arrowhead-std)" />
      <line x1="148" y1="72" x2="163" y2="72" stroke="var(--ink-mute)" strokeWidth="1"
        markerEnd="url(#arrowhead-std)" />
      <line x1="187" y1="68" x2="196" y2="68" stroke="var(--ink-mute)" strokeWidth="1"
        markerEnd="url(#arrowhead-std)" />
      <line x1="213" y1="68" x2="226" y2="68" stroke="var(--ink-mute)" strokeWidth="1"
        markerEnd="url(#arrowhead-std)" />

      {/* Arrow out of dense block */}
      <line x1="270" y1="78" x2="286" y2="78" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-std)" />

      {/* ×3 label */}
      <text x="275" y="74" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">×3</text>

      {/* Transition Layer */}
      <rect x="286" y="66" width="44" height="24" rx="4"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="308" y="76" textAnchor="middle" fontSize="7" fill="var(--ink-soft)" fontFamily="monospace">Trans.</text>
      <text x="308" y="85" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">1×1+pool</text>

      {/* Arrow Transition → GAP */}
      <line x1="330" y1="78" x2="346" y2="78" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-std)" />

      {/* GAP circle */}
      <circle cx="362" cy="78" r="15" fill="var(--surface-soft)"
        stroke="var(--border)" strokeWidth="1" />
      <text x="362" y="75" textAnchor="middle" fontSize="6.5" fill="var(--ink-soft)" fontFamily="monospace">GAP</text>
      <text x="362" y="84" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">→1D</text>

      {/* Arrow GAP → Softmax */}
      <line x1="377" y1="78" x2="393" y2="78" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-std)" />

      {/* Softmax / FC box */}
      <rect x="393" y="64" width="68" height="28" rx="4"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="427" y="76" textAnchor="middle" fontSize="7.5" fill="var(--ink-soft)" fontFamily="monospace">FC</text>
      <text x="427" y="86" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">Softmax</text>
    </svg>
  );
}

// ─── Our Triple Attention DenseNet SVG ───────────────────────────────────────

function OurDenseNetSVG() {
  return (
    <svg viewBox="0 0 470 175" className="w-full" role="img" aria-label="Triple Attention DenseNet architecture">
      <defs>
        <marker id="arrowhead-our" markerWidth="6" markerHeight="5"
          refX="5" refY="2.5" orient="auto">
          <polygon points="0 0, 6 2.5, 0 5" fill="var(--ink-mute)" />
        </marker>
      </defs>

      {/* Input image */}
      <rect x="6" y="72" width="28" height="28" rx="4"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="20" y="85" textAnchor="middle" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">224²</text>
      <text x="20" y="95" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">×3ch</text>

      {/* Arrow → backbone */}
      <line x1="34" y1="86" x2="52" y2="86" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />

      {/* DenseNet121 Backbone */}
      <rect x="52" y="54" width="72" height="64" rx="6"
        fill="var(--pos-soft)" stroke="var(--pos-deep)" strokeWidth="1.2" />
      <text x="88" y="80" textAnchor="middle" fontSize="8" fill="var(--pos-deep)" fontFamily="monospace" fontWeight="600">Dense</text>
      <text x="88" y="90" textAnchor="middle" fontSize="8" fill="var(--pos-deep)" fontFamily="monospace" fontWeight="600">Net121</text>
      <text x="88" y="108" textAnchor="middle" fontSize="7" fill="var(--pos)" fontFamily="monospace">1024-dim</text>
      <text x="88" y="116" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">7×7 feat</text>

      {/* Three split arrows from backbone */}
      {/* Main line out */}
      <line x1="124" y1="86" x2="138" y2="86" stroke="var(--ink-mute)" strokeWidth="1.2" />
      {/* Up branch to pos */}
      <line x1="138" y1="86" x2="138" y2="36" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="138" y1="36" x2="154" y2="36" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />
      {/* Middle branch to chan */}
      <line x1="138" y1="86" x2="154" y2="86" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />
      {/* Down branch to type */}
      <line x1="138" y1="86" x2="138" y2="136" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="138" y1="136" x2="154" y2="136" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />

      {/* Pos Attention branch */}
      <rect x="154" y="20" width="76" height="32" rx="5"
        fill="var(--pos-soft)" stroke="var(--pos)" strokeWidth="1.2" />
      <text x="192" y="34" textAnchor="middle" fontSize="7.5" fill="var(--pos-deep)" fontFamily="monospace" fontWeight="700">Position Attn</text>
      <text x="192" y="44" textAnchor="middle" fontSize="6.5" fill="var(--pos)" fontFamily="monospace">Softmax(QKᵀ/√d)V</text>

      {/* Chan Attention branch */}
      <rect x="154" y="70" width="76" height="32" rx="5"
        fill="var(--chan-soft)" stroke="var(--chan)" strokeWidth="1.2" />
      <text x="192" y="84" textAnchor="middle" fontSize="7.5" fill="var(--chan-deep)" fontFamily="monospace" fontWeight="700">Channel Attn</text>
      <text x="192" y="94" textAnchor="middle" fontSize="6.5" fill="var(--chan)" fontFamily="monospace">Softmax(CCᵀ)·x</text>

      {/* Type Attention branch */}
      <rect x="154" y="120" width="76" height="32" rx="5"
        fill="var(--type-soft)" stroke="var(--type-deep)" strokeWidth="1.2" />
      <text x="192" y="134" textAnchor="middle" fontSize="7.5" fill="var(--type-deep)" fontFamily="monospace" fontWeight="700">Type Attn</text>
      <text x="192" y="144" textAnchor="middle" fontSize="6.5" fill="var(--type)" fontFamily="monospace">MS-Conv + SE gate</text>

      {/* Arrows from branches → fusion */}
      <line x1="230" y1="36" x2="244" y2="36" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="244" y1="36" x2="244" y2="86" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="230" y1="86" x2="244" y2="86" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="230" y1="136" x2="244" y2="136" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="244" y1="86" x2="244" y2="136" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="244" y1="86" x2="258" y2="86" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />

      {/* Fusion block */}
      <rect x="258" y="66" width="60" height="40" rx="5"
        fill="var(--fusion-soft)" stroke="var(--fusion)" strokeWidth="1.2" />
      <text x="288" y="84" textAnchor="middle" fontSize="8" fill="var(--fusion)" fontFamily="monospace" fontWeight="700">Fusion</text>
      <text x="288" y="94" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">(p+c+t)/3</text>
      <text x="288" y="104" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">+GAP</text>

      {/* Arrows Fusion → 3 heads */}
      <line x1="318" y1="86" x2="332" y2="86" stroke="var(--ink-mute)" strokeWidth="1.2" />
      <line x1="332" y1="86" x2="332" y2="30" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="332" y1="86" x2="332" y2="142" stroke="var(--ink-mute)" strokeWidth="1" />
      <line x1="332" y1="30" x2="348" y2="30" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />
      <line x1="332" y1="86" x2="348" y2="86" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />
      <line x1="332" y1="142" x2="348" y2="142" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-our)" />

      {/* Head 1: Type */}
      <rect x="348" y="16" width="110" height="28" rx="4"
        fill="#FEF0EB" stroke="var(--head-type)" strokeWidth="1.2" />
      <text x="403" y="28" textAnchor="middle" fontSize="7.5" fill="var(--head-type)" fontFamily="monospace" fontWeight="700">Head 1 · BSS Type</text>
      <text x="403" y="38" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">512→256→7 · FocalLoss</text>

      {/* Head 2: Shape */}
      <rect x="348" y="72" width="110" height="28" rx="4"
        fill="#EEF2FE" stroke="var(--head-shape)" strokeWidth="1.2" />
      <text x="403" y="84" textAnchor="middle" fontSize="7.5" fill="var(--head-shape)" fontFamily="monospace" fontWeight="700">Head 2 · Shape</text>
      <text x="403" y="94" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">256→4 · CrossEntropy</text>

      {/* Head 3: Color */}
      <rect x="348" y="128" width="110" height="28" rx="4"
        fill="#FEF0F5" stroke="var(--head-color)" strokeWidth="1.2" />
      <text x="403" y="140" textAnchor="middle" fontSize="7.5" fill="var(--head-color)" fontFamily="monospace" fontWeight="700">Head 3 · Color</text>
      <text x="403" y="150" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">128→2 · CrossEntropy</text>
    </svg>
  );
}

// ─── Public component ────────────────────────────────────────────────────────

export function DenseNetArchSection() {
  return (
    <section className="mx-auto max-w-[1060px] px-5 py-10">
      <div className="mb-6">
        <span className="font-mono text-[10px] uppercase tracking-[.1em]"
          style={{ color: "var(--ink-mute)" }}>Architecture Overview</span>
        <h2 className="font-display mt-1 text-xl font-bold" style={{ color: "var(--foreground)" }}>
          DenseNet — Standard vs. Our Modification
        </h2>
        <p className="mt-1.5 max-w-2xl text-sm" style={{ color: "var(--ink-soft)" }}>
          DenseNet solves the vanishing-gradient problem by connecting every layer directly to all
          subsequent layers. We replace the classification head with three parallel attention
          modules and three multi-task outputs.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {/* ── Standard DenseNet ── */}
        <div className="rounded-2xl border p-5" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <div className="mb-3 flex items-center gap-2">
            <span className="rounded-full px-2.5 py-0.5 font-mono text-[10px] font-semibold"
              style={{ background: "var(--surface-soft)", border: "1px solid var(--border)", color: "var(--ink-soft)" }}>
              Standard
            </span>
            <span className="font-display text-[13px] font-bold" style={{ color: "var(--foreground)" }}>
              DenseNet (DenseNet-BC)
            </span>
          </div>

          <div className="rounded-xl p-3 mb-4" style={{ background: "var(--surface-soft)" }}>
            <StandardDenseNetSVG />
          </div>

          <ul className="space-y-2">
            {[
              ["Dense connectivity", "Each layer x_l receives feature maps from ALL preceding layers: x_l = H_l([x₀,…,x_{l-1}])"],
              ["Growth rate k", "Each layer adds only k new feature maps — small k keeps the model compact"],
              ["Transition layers", "1×1 conv (halves channels) + 2×2 avg pool between dense blocks"],
              ["Classification head", "Global average pool → single fully-connected layer → Softmax(N classes)"],
            ].map(([title, body]) => (
              <li key={title as string} className="flex gap-2.5 text-sm">
                <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: "var(--ink-mute)", marginTop: "0.45rem" }} />
                <span>
                  <span className="font-semibold" style={{ color: "var(--foreground)" }}>{title}: </span>
                  <span style={{ color: "var(--ink-soft)" }}>{body}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* ── Our Triple Attention DenseNet ── */}
        <div className="rounded-2xl border p-5" style={{ borderColor: "var(--pos)", background: "var(--surface)" }}>
          <div className="mb-3 flex items-center gap-2">
            <span className="rounded-full px-2.5 py-0.5 font-mono text-[10px] font-semibold"
              style={{ background: "var(--pos-soft)", border: "1px solid var(--pos)", color: "var(--pos-deep)" }}>
              Our Model
            </span>
            <span className="font-display text-[13px] font-bold" style={{ color: "var(--foreground)" }}>
              StoolNetTriple · Triple Attention
            </span>
          </div>

          <div className="rounded-xl p-3 mb-4" style={{ background: "var(--surface-soft)" }}>
            <OurDenseNetSVG />
          </div>

          <ul className="space-y-2">
            {[
              ["Pretrained backbone", "DenseNet121 (ImageNet) used as frozen feature extractor — 1024-dim, 7×7 spatial output"],
              ["Position Attention", "Spatial self-attention via Q/K/V projections + residual γ=0.1 — captures where features matter"],
              ["Channel Attention", "C×C correlation matrix reweights feature channels by co-occurrence patterns"],
              ["Type Attention", "Multi-scale (3×3/5×5/7×7) + Squeeze-Excite + sigmoid ordinal gate for BSS continuity"],
              ["Multi-task heads", "Three independent classifiers share the fused attention output (BSS type, shape, color)"],
            ].map(([title, body]) => (
              <li key={title as string} className="flex gap-2.5 text-sm">
                <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: "var(--pos)", marginTop: "0.45rem" }} />
                <span>
                  <span className="font-semibold" style={{ color: "var(--foreground)" }}>{title}: </span>
                  <span style={{ color: "var(--ink-soft)" }}>{body}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Key difference callout */}
      <div className="mt-5 rounded-xl border-l-4 px-5 py-4"
        style={{ borderColor: "var(--fusion)", background: "var(--fusion-soft)" }}>
        <span className="font-display text-[12px] font-bold" style={{ color: "var(--fusion)" }}>
          Key innovation:
        </span>
        <span className="ml-2 text-sm" style={{ color: "var(--ink-soft)" }}>
          We replace the single classification head with <strong>3 parallel attention modules</strong> (each
          with a learnable γ residual starting at 0.1) that specialise in spatial, channel, and
          texture-ordinal information — then fuse their outputs to feed three task-specific heads
          simultaneously.
        </span>
      </div>
    </section>
  );
}
