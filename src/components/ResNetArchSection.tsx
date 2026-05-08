// Architecture overview: Standard ResNet-50 → Our 5-Fold CV ResNet-50

// ─── Standard ResNet-50 SVG ──────────────────────────────────────────────────

function StandardResNetSVG() {
  const stages = [
    { label: "Conv\n7×7", sub: "64ch\n112²", x: 90,  color: "#5B8FD0", w: 36, h: 52 },
    { label: "Stage\n2",   sub: "256ch\n56²",  x: 148, color: "#4A9E74", w: 36, h: 52 },
    { label: "Stage\n3",   sub: "512ch\n28²",  x: 206, color: "#9A6BC0", w: 36, h: 52 },
    { label: "Stage\n4",   sub: "1024\n14²",   x: 264, color: "#C09030", w: 36, h: 52 },
    { label: "Stage\n5",   sub: "2048\n7²",    x: 322, color: "#C05040", w: 36, h: 52 },
  ];

  const blockCounts = [1, 3, 4, 6, 3];

  return (
    <svg viewBox="0 0 470 175" className="w-full" role="img" aria-label="Standard ResNet-50 architecture">
      <defs>
        <marker id="arrowhead-rn" markerWidth="6" markerHeight="5"
          refX="5" refY="2.5" orient="auto">
          <polygon points="0 0, 6 2.5, 0 5" fill="var(--ink-mute)" />
        </marker>
      </defs>

      {/* ─── Bottleneck Block detail (top-left inset) ─── */}
      <rect x="4" y="8" width="78" height="130" rx="6"
        fill="none" stroke="var(--border)" strokeWidth="1" strokeDasharray="3 2" />
      <text x="43" y="18" textAnchor="middle" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">Bottleneck Block</text>

      {/* x input */}
      <text x="43" y="32" textAnchor="middle" fontSize="8" fill="var(--ink-soft)" fontFamily="monospace">x</text>
      <line x1="43" y1="34" x2="43" y2="42" stroke="var(--ink-mute)" strokeWidth="1.2"
        markerEnd="url(#arrowhead-rn)" />

      {/* 1×1 conv reduce */}
      <rect x="15" y="42" width="56" height="18" rx="3"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="43" y="52" textAnchor="middle" fontSize="6.5" fill="var(--ink-soft)" fontFamily="monospace">1×1 conv (reduce)</text>
      <text x="43" y="60" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">e.g. 1024→256</text>
      <line x1="43" y1="60" x2="43" y2="68" stroke="var(--ink-mute)" strokeWidth="1.2"
        markerEnd="url(#arrowhead-rn)" />

      {/* 3×3 conv */}
      <rect x="15" y="68" width="56" height="18" rx="3"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="43" y="79" textAnchor="middle" fontSize="6.5" fill="var(--ink-soft)" fontFamily="monospace">3×3 conv (process)</text>
      <line x1="43" y1="86" x2="43" y2="94" stroke="var(--ink-mute)" strokeWidth="1.2"
        markerEnd="url(#arrowhead-rn)" />

      {/* 1×1 conv expand */}
      <rect x="15" y="94" width="56" height="18" rx="3"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="43" y="104" textAnchor="middle" fontSize="6.5" fill="var(--ink-soft)" fontFamily="monospace">1×1 conv (expand)</text>
      <text x="43" y="112" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">256→1024</text>

      {/* Skip connection arc */}
      <path d="M 43 34 C 75 34, 75 112, 43 112" fill="none"
        stroke="#5B8FD0" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x="76" y="75" fontSize="6.5" fill="#5B8FD0" fontFamily="monospace">x</text>

      {/* Add node */}
      <line x1="43" y1="112" x2="43" y2="120" stroke="var(--ink-mute)" strokeWidth="1.2" />
      <circle cx="43" cy="124" r="7" fill="var(--surface-soft)"
        stroke="var(--border)" strokeWidth="1.2" />
      <text x="43" y="128" textAnchor="middle" fontSize="9" fill="var(--ink-soft)" fontFamily="monospace">+</text>
      <line x1="43" y1="131" x2="43" y2="139" stroke="var(--ink-mute)" strokeWidth="1.2"
        markerEnd="url(#arrowhead-rn)" />
      <text x="43" y="147" textAnchor="middle" fontSize="7.5" fill="var(--ink-soft)" fontFamily="monospace">F(x)+x</text>

      {/* ─── Main pipeline (right side) ─── */}

      {/* Input image */}
      <rect x="90" y="62" width="36" height="36" rx="5"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="108" y="78" textAnchor="middle" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">224²</text>
      <text x="108" y="88" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">×3ch</text>

      {/* Pipeline stages */}
      {stages.map((s, i) => {
        const prevX = i === 0 ? 90 + 36 : stages[i - 1].x + stages[i - 1].w;
        const arrowX1 = prevX;
        const arrowX2 = s.x;
        const midY = 80;
        return (
          <g key={s.label}>
            {/* Arrow between blocks */}
            <line x1={arrowX1} y1={midY} x2={arrowX2 - 1} y2={midY}
              stroke="var(--ink-mute)" strokeWidth="1.2" markerEnd="url(#arrowhead-rn)" />

            {/* Stage block */}
            <rect x={s.x} y={midY - s.h / 2} width={s.w} height={s.h} rx="4"
              fill={s.color} opacity={0.15} />
            <rect x={s.x} y={midY - s.h / 2} width={s.w} height={s.h} rx="4"
              fill="none" stroke={s.color} strokeWidth="1.5" />
            {/* Block count indicator dots */}
            <g>
              {Array.from({ length: Math.min(blockCounts[i], 6) }, (_, j) => (
                <rect key={j} x={s.x + 4 + j * 4} y={midY - s.h / 2 + 4} width={2.5} height={2.5}
                  rx="0.5" fill={s.color} opacity={0.7} />
              ))}
            </g>
            {/* Labels */}
            {s.label.split("\n").map((line, li) => (
              <text key={li} x={s.x + s.w / 2} y={midY - 10 + li * 10}
                textAnchor="middle" fontSize="7" fill={s.color}
                fontFamily="monospace" fontWeight="700">{line}</text>
            ))}
            {s.sub.split("\n").map((line, li) => (
              <text key={li} x={s.x + s.w / 2} y={midY + 14 + li * 9}
                textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">{line}</text>
            ))}
            {/* Block count label */}
            <text x={s.x + s.w / 2} y={midY + s.h / 2 + 11}
              textAnchor="middle" fontSize="6.5" fill={s.color} fontFamily="monospace">
              ×{blockCounts[i]}
            </text>
          </g>
        );
      })}

      {/* Arrow Stage 5 → GAP */}
      <line x1={322 + 36} y1={80} x2={375} y2={80} stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-rn)" />

      {/* GAP */}
      <circle cx="388" cy="80" r="14" fill="var(--surface-soft)"
        stroke="var(--border)" strokeWidth="1" />
      <text x="388" y="77" textAnchor="middle" fontSize="6.5" fill="var(--ink-soft)" fontFamily="monospace">GAP</text>
      <text x="388" y="86" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">2048</text>

      {/* Arrow GAP → FC */}
      <line x1="402" y1="80" x2="416" y2="80" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-rn)" />

      {/* FC + Softmax */}
      <rect x="416" y="66" width="50" height="28" rx="4"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="441" y="78" textAnchor="middle" fontSize="7" fill="var(--ink-soft)" fontFamily="monospace">FC</text>
      <text x="441" y="88" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">Softmax</text>

      {/* 1000 classes label */}
      <text x="441" y="103" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">1000 cls</text>

      {/* Total layers counter */}
      <text x="280" y="155" textAnchor="middle" fontSize="8" fill="var(--ink-mute)" fontFamily="monospace">
        50 layers total: 1 + (3+4+6+3)×3 + 1 = 50
      </text>
    </svg>
  );
}

// ─── Our ResNet-50 5-Fold CV SVG ─────────────────────────────────────────────

function OurResNetSVG() {
  const folds = [
    { acc: 76.9, color: "#1D9E75" },
    { acc: 53.8, color: "#C07A0A" },
    { acc: 61.5, color: "#C07A0A" },
    { acc: 58.3, color: "#C07A0A" },
    { acc: 50.0, color: "#B83820" },
  ];

  const trainW = 78;
  const totalW = 110;

  return (
    <svg viewBox="0 0 470 175" className="w-full" role="img" aria-label="5-Fold Cross-Validation ResNet-50">
      <defs>
        <marker id="arrowhead-cv" markerWidth="6" markerHeight="5"
          refX="5" refY="2.5" orient="auto">
          <polygon points="0 0, 6 2.5, 0 5" fill="var(--ink-mute)" />
        </marker>
      </defs>

      {/* ─── Training side ─── */}
      <text x="68" y="12" textAnchor="middle" fontSize="8.5" fill="var(--ink-soft)"
        fontFamily="monospace" fontWeight="700">TRAINING (5-Fold CV)</text>

      {/* Dataset splits */}
      {folds.map((f, i) => {
        const y = 20 + i * 24;
        const valX = 8 + trainW;
        const valW = totalW - trainW;
        return (
          <g key={i}>
            {/* Fold label */}
            <text x="4" y={y + 14} fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">F{i + 1}</text>
            {/* Train segment */}
            <rect x="16" y={y + 4} width={trainW} height="16" rx="2"
              fill="#1D9E75" opacity={0.25} />
            {/* Val segment */}
            <rect x={valX + 16} y={y + 4} width={valW} height="16" rx="2"
              fill="#C07A0A" opacity={0.35} />
            {/* Val highlight for this fold */}
            <rect x={valX + 16 - (i * (valW / 5))} y={y + 4} width={valW / 5} height="16" rx="2"
              fill="#C07A0A" opacity={0.7} />
            {/* Accuracy label */}
            <text x={totalW + 20} y={y + 14} fontSize="6.5" fill={f.color}
              fontFamily="monospace" fontWeight="700">{f.acc}%</text>
          </g>
        );
      })}

      {/* Legend */}
      <rect x="16" y="143" width="8" height="8" rx="1" fill="#1D9E75" opacity={0.4} />
      <text x="26" y="151" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">train (50-51)</text>
      <rect x="70" y="143" width="8" height="8" rx="1" fill="#C07A0A" opacity={0.55} />
      <text x="80" y="151" fontSize="6.5" fill="var(--ink-mute)" fontFamily="monospace">val (12-13)</text>

      {/* ResNet50 model in center */}
      <rect x="140" y="32" width="68" height="110" rx="8"
        fill="#EEF4FF" stroke="#5B8FD0" strokeWidth="1.5" />
      <text x="174" y="68" textAnchor="middle" fontSize="8.5" fill="#3B68B0"
        fontFamily="monospace" fontWeight="700">ResNet</text>
      <text x="174" y="80" textAnchor="middle" fontSize="8.5" fill="#3B68B0"
        fontFamily="monospace" fontWeight="700">50</text>
      <text x="174" y="100" textAnchor="middle" fontSize="7" fill="var(--ink-mute)"
        fontFamily="monospace">pretrained</text>
      <text x="174" y="110" textAnchor="middle" fontSize="7" fill="var(--ink-mute)"
        fontFamily="monospace">ImageNet</text>
      <text x="174" y="126" textAnchor="middle" fontSize="6.5" fill="#3B68B0"
        fontFamily="monospace">+ Classifier</text>
      <text x="174" y="135" textAnchor="middle" fontSize="6" fill="var(--ink-mute)"
        fontFamily="monospace">Drop→512→256→7</text>

      {/* Arrows fold → model */}
      {folds.map((_, i) => {
        const y = 20 + i * 24 + 12;
        return (
          <line key={i} x1="132" y1={y} x2="140" y2={87}
            stroke="var(--ink-mute)" strokeWidth="0.8" opacity={0.5}
            markerEnd="url(#arrowhead-cv)" />
        );
      })}

      {/* Divider */}
      <line x1="225" y1="20" x2="225" y2="160" stroke="var(--border)"
        strokeWidth="1" strokeDasharray="4 3" />
      <text x="225" y="168" textAnchor="middle" fontSize="7.5" fill="var(--ink-mute)"
        fontFamily="monospace">inference →</text>

      {/* ─── Inference side ─── */}
      <text x="352" y="12" textAnchor="middle" fontSize="8.5" fill="var(--ink-soft)"
        fontFamily="monospace" fontWeight="700">INFERENCE (TTA + Ensemble)</text>

      {/* Input image */}
      <rect x="230" y="55" width="30" height="30" rx="4"
        fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="1" />
      <text x="245" y="68" textAnchor="middle" fontSize="7" fill="var(--ink-mute)" fontFamily="monospace">img</text>
      <text x="245" y="78" textAnchor="middle" fontSize="6" fill="var(--ink-mute)" fontFamily="monospace">input</text>

      {/* TTA augmentations */}
      <text x="278" y="35" textAnchor="middle" fontSize="7" fill="var(--ink-mute)"
        fontFamily="monospace">×8 TTA</text>
      {["orig", "H↔", "V↕", "R+15°", "R−15°", "H+V", "R+H", "R+V"].map((lbl, i) => {
        const row = Math.floor(i / 2);
        const col = i % 2;
        const tx = 252 + col * 28;
        const ty = 40 + row * 20;
        return (
          <g key={i}>
            <rect x={tx} y={ty} width={24} height={14} rx="2"
              fill="var(--surface-soft)" stroke="var(--border)" strokeWidth="0.8" />
            <text x={tx + 12} y={ty + 9} textAnchor="middle" fontSize="5.5"
              fill="var(--ink-soft)" fontFamily="monospace">{lbl}</text>
          </g>
        );
      })}

      {/* Arrow TTA → 5 models */}
      <line x1="310" y1="70" x2="324" y2="70" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-cv)" />

      {/* 5 fold models */}
      {folds.map((f, i) => {
        const y = 25 + i * 26;
        return (
          <g key={i}>
            <rect x="324" y={y} width="52" height="18" rx="3"
              fill={f.color} opacity={0.1} />
            <rect x="324" y={y} width="52" height="18" rx="3"
              fill="none" stroke={f.color} strokeWidth="1" />
            <text x="350" y={y + 8} textAnchor="middle" fontSize="6.5"
              fill={f.color} fontFamily="monospace" fontWeight="600">Fold {i + 1}</text>
            <text x="350" y={y + 15} textAnchor="middle" fontSize="5.5"
              fill="var(--ink-mute)" fontFamily="monospace">{f.acc}% val</text>
            {/* Arrow to ensemble */}
            <line x1="376" y1={y + 9} x2="392" y2="86"
              stroke="var(--ink-mute)" strokeWidth="0.8" opacity={0.5}
              markerEnd="url(#arrowhead-cv)" />
          </g>
        );
      })}

      {/* Ensemble average */}
      <rect x="392" y="62" width="72" height="48" rx="6"
        fill="#E8F5EF" stroke="#1D9E75" strokeWidth="1.5" />
      <text x="428" y="80" textAnchor="middle" fontSize="7" fill="#1D7A5C"
        fontFamily="monospace" fontWeight="700">Ensemble</text>
      <text x="428" y="90" textAnchor="middle" fontSize="6.5" fill="#1D7A5C"
        fontFamily="monospace">avg probs</text>
      <text x="428" y="100" textAnchor="middle" fontSize="6.5" fill="var(--ink-mute)"
        fontFamily="monospace">60.1% mean</text>

      {/* Arrow → prediction */}
      <line x1="428" y1="110" x2="428" y2="124" stroke="var(--ink-mute)"
        strokeWidth="1.2" markerEnd="url(#arrowhead-cv)" />

      {/* Prediction output */}
      <rect x="392" y="124" width="72" height="26" rx="5"
        fill="#1D9E75" />
      <text x="428" y="136" textAnchor="middle" fontSize="7.5" fill="white"
        fontFamily="monospace" fontWeight="700">BSS Type</text>
      <text x="428" y="145" textAnchor="middle" fontSize="6.5" fill="rgba(255,255,255,0.75)"
        fontFamily="monospace">T1–T7 softmax</text>
    </svg>
  );
}

// ─── Public component ────────────────────────────────────────────────────────

export function ResNetArchSection() {
  return (
    <section className="mx-auto max-w-[1060px] px-5 py-10">
      <div className="mb-6">
        <span className="font-mono text-[10px] uppercase tracking-[.1em]"
          style={{ color: "var(--ink-mute)" }}>Architecture Overview</span>
        <h2 className="font-display mt-1 text-xl font-bold" style={{ color: "var(--foreground)" }}>
          ResNet-50 — Standard vs. Our 5-Fold CV Pipeline
        </h2>
        <p className="mt-1.5 max-w-2xl text-sm" style={{ color: "var(--ink-soft)" }}>
          ResNet-50 introduced skip connections that allow gradients to flow through very deep networks
          without vanishing. We adapt it with a custom classifier head, 5-fold cross-validation on a
          small dataset, and test-time augmentation ensemble inference.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {/* ── Standard ResNet-50 ── */}
        <div className="rounded-2xl border p-5" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <div className="mb-3 flex items-center gap-2">
            <span className="rounded-full px-2.5 py-0.5 font-mono text-[10px] font-semibold"
              style={{ background: "var(--surface-soft)", border: "1px solid var(--border)", color: "var(--ink-soft)" }}>
              Standard
            </span>
            <span className="font-display text-[13px] font-bold" style={{ color: "var(--foreground)" }}>
              ResNet-50 (He et al., 2015)
            </span>
          </div>

          <div className="rounded-xl p-3 mb-4" style={{ background: "var(--surface-soft)" }}>
            <StandardResNetSVG />
          </div>

          <ul className="space-y-2">
            {[
              ["Skip connections", "F(x) + x — each block learns the residual change, not a full mapping. Identity shortcuts preserve information across depth"],
              ["Bottleneck design", "1×1 reduce → 3×3 process → 1×1 expand. Cuts parameters by 4× vs naïve 3×3 stacks while maintaining capacity"],
              ["5 stages", "Conv7×7 → Stage2 (256ch) → Stage3 (512ch) → Stage4 (1024ch) → Stage5 (2048ch). Spatial dims halve each stage"],
              ["Classification", "Global average pool over 7×7 feature map → 2048-dim vector → FC → Softmax(1000)"],
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

        {/* ── Our 5-Fold CV ResNet-50 ── */}
        <div className="rounded-2xl border p-5" style={{ borderColor: "#1D9E75", background: "var(--surface)" }}>
          <div className="mb-3 flex items-center gap-2">
            <span className="rounded-full px-2.5 py-0.5 font-mono text-[10px] font-semibold"
              style={{ background: "#E8F5EF", border: "1px solid #1D9E75", color: "#1D7A5C" }}>
              Our Model
            </span>
            <span className="font-display text-[13px] font-bold" style={{ color: "var(--foreground)" }}>
              ResNet50 · 5-Fold CV + TTA Ensemble
            </span>
          </div>

          <div className="rounded-xl p-3 mb-4" style={{ background: "var(--surface-soft)" }}>
            <OurResNetSVG />
          </div>

          <ul className="space-y-2">
            {[
              ["Small dataset (n=63)", "5-fold stratified CV avoids overfitting. Each fold: ~50 train + ~13 val. All 5 models kept for ensemble"],
              ["Custom classifier head", "Frozen ResNet50 backbone + Dropout(0.5) → 512 → ReLU+BN → Dropout(0.3) → 256 → 7-class Softmax"],
              ["Training augmentations", "MixUp α=0.3, CutMix p=0.5, EMA decay=0.999, CosineWarmRestart, early stop patience=12"],
              ["TTA × 8", "At inference: original + H-flip + V-flip + ±15° rotation + H+V + R+H + R+V → average 8 softmax vectors"],
              ["Ensemble", "5 fold models × 8 TTA = 40 forward passes per image. Probability average → argmax → BSS type T1–T7"],
            ].map(([title, body]) => (
              <li key={title as string} className="flex gap-2.5 text-sm">
                <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ background: "#1D9E75", marginTop: "0.45rem" }} />
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
        style={{ borderColor: "#1D9E75", background: "#E8F5EF" }}>
        <span className="font-display text-[12px] font-bold" style={{ color: "#1D7A5C" }}>
          Key adaptation:
        </span>
        <span className="ml-2 text-sm" style={{ color: "var(--ink-soft)" }}>
          Standard ResNet-50 is trained end-to-end on millions of images. Our version{" "}
          <strong>fine-tunes only the classifier head</strong> on a 63-image clinical dataset using
          5-fold CV to maximise data utilisation, then runs <strong>8-augmentation TTA</strong> at
          inference time across all 5 fold models to reduce prediction variance.
        </span>
      </div>
    </section>
  );
}
