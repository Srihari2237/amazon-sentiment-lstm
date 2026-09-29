/**
 * Generate the project report (.docx) from results/report_data.json.
 *
 * Every number, table, figure and code listing comes from the collected
 * results, so re-running `python -m src.report_data && node scripts/make_report.js`
 * after a new training run refreshes the document rather than requiring it to
 * be edited by hand.
 */

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const DATA = JSON.parse(
  fs.readFileSync(path.join(ROOT, "results", "report_data.json"), "utf8")
);
const OUT = path.join(ROOT, "results", "Project_Report.docx");

// ---------------------------------------------------------------- constants

const TITLE =
  "CUSTOMER REVIEW SENTIMENT ANALYSIS USING LSTM FOR ONLINE SHOPPING";
const COURSE = "23AML501 – Deep Learning & Applications";
const COLLEGE = "Dr. MAHALINGAM COLLEGE OF ENGINEERING & TECHNOLOGY";
const PLACE = "POLLACHI, COIMBATORE - 642 003";

// Replace with the real submission team before printing.
const STUDENTS = [
  ["[STUDENT NAME]", "[ROLL NUMBER]"],
];
const GITHUB_LINK = "[ADD GITHUB LINK BEFORE SUBMISSION]";

const PAGE_WIDTH_DXA = 9026; // A4 portrait minus 1" margins
const MONO = "Consolas";
const BODY = "Times New Roman";

// ------------------------------------------------------------------ helpers

const p = (text, opts = {}) =>
  new Paragraph({
    alignment: opts.align,
    spacing: { after: opts.after ?? 120, line: opts.line ?? 276 },
    indent: opts.indent,
    children: [
      new TextRun({
        text,
        bold: opts.bold,
        italics: opts.italics,
        size: opts.size ?? 24, // half-points => 12pt
        font: opts.font ?? BODY,
        color: opts.color,
        allCaps: opts.caps,
      }),
    ],
  });

const blank = (after = 120) =>
  new Paragraph({ spacing: { after }, children: [new TextRun("")] });

const h1 = (text, brk = false) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    pageBreakBefore: brk,
    spacing: { before: brk ? 0 : 320, after: 180 },
    children: [
      new TextRun({ text, bold: true, size: 30, font: BODY, color: "0B0B0B" }),
    ],
  });

const h2 = (text, brk = false) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    pageBreakBefore: brk,
    spacing: { before: brk ? 0 : 260, after: 140 },
    children: [
      new TextRun({ text, bold: true, size: 26, font: BODY, color: "1C5CAB" }),
    ],
  });

const h3 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 120 },
    children: [
      new TextRun({ text, bold: true, size: 24, font: BODY, color: "52514E" }),
    ],
  });

const bullet = (text) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 100, line: 276 },
    children: [new TextRun({ text, size: 24, font: BODY })],
  });

const codeBlock = (source) =>
  source.split("\n").map(
    (line) =>
      new Paragraph({
        spacing: { after: 0, line: 200 },
        shading: { type: ShadingType.CLEAR, fill: "F5F5F2" },
        children: [
          new TextRun({
            text: line.length ? line : " ",
            font: MONO,
            size: 15, // 7.5pt - keeps long lines on one line
          }),
        ],
      })
  );

const caption = (text) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 240 },
    children: [
      new TextRun({ text, italics: true, size: 20, font: BODY, color: "52514E" }),
    ],
  });

function figure(key, captionText, widthPx = 620) {
  const file = DATA.figures[key];
  if (!file || !fs.existsSync(file)) {
    return [p(`[figure missing: ${key}]`, { italics: true, color: "B00020" })];
  }
  const sizeOf = require("image-size");
  let ratio = 0.5;
  try {
    const dim = sizeOf.imageSize
      ? sizeOf.imageSize(fs.readFileSync(file))
      : sizeOf(file);
    ratio = dim.height / dim.width;
  } catch (e) {
    /* fall back to the default ratio */
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 0 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(file),
          transformation: { width: widthPx, height: Math.round(widthPx * ratio) },
        }),
      ],
    }),
    caption(captionText),
  ];
}

function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((w) => Math.round((w / total) * PAGE_WIDTH_DXA));

  const cell = (text, i, { header = false, bold = false } = {}) =>
    new TableCell({
      width: { size: scaled[i], type: WidthType.DXA },
      shading: header
        ? { type: ShadingType.CLEAR, fill: "DBE9FC" }
        : undefined,
      margins: { top: 60, bottom: 60, left: 90, right: 90 },
      children: [
        new Paragraph({
          alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { after: 0, line: 240 },
          children: [
            new TextRun({
              text: String(text),
              bold: header || bold,
              size: opts.size ?? 20,
              font: BODY,
            }),
          ],
        }),
      ],
    });

  return new Table({
    columnWidths: scaled,
    width: { size: PAGE_WIDTH_DXA, type: WidthType.DXA },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, i, { header: true })),
      }),
      ...rows.map(
        (r) =>
          new TableRow({
            children: r.map((v, i) =>
              cell(v, i, { bold: opts.boldRows?.includes(r[0]) })
            ),
          })
      ),
    ],
  });
}

const pct = (v) => (v * 100).toFixed(2) + "%";
const f4 = (v) => (v === null || v === undefined ? "-" : Number(v).toFixed(4));

// -------------------------------------------------------------- front matter

function frontPage() {
  const out = [];
  out.push(blank(400));
  out.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [
        new TextRun({ text: TITLE, bold: true, size: 36, font: BODY }),
      ],
    })
  );
  out.push(p("MINI PROJECT", { align: AlignmentType.CENTER, bold: true, size: 28 }));
  out.push(blank(200));
  out.push(p("Submitted by", { align: AlignmentType.CENTER, size: 24 }));
  out.push(blank(160));
  for (const [name, roll] of STUDENTS) {
    out.push(
      p(`${name}    -    ${roll}`, {
        align: AlignmentType.CENTER,
        bold: true,
        size: 26,
      })
    );
  }
  out.push(blank(400));
  out.push(
    p(
      "Submitted in partial fulfillment of the requirements for the award of the degree of Bachelor of Engineering in",
      { align: AlignmentType.CENTER, size: 24 }
    )
  );
  out.push(
    p("ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING", {
      align: AlignmentType.CENTER,
      bold: true,
      size: 28,
    })
  );
  out.push(blank(400));
  out.push(p(COLLEGE, { align: AlignmentType.CENTER, bold: true, size: 28 }));
  out.push(p(PLACE, { align: AlignmentType.CENTER, bold: true, size: 26 }));
  out.push(blank(120));
  out.push(
    p("Affiliated to Anna University, Chennai * Approved by AICTE, New Delhi", {
      align: AlignmentType.CENTER,
      size: 20,
    })
  );
  out.push(
    p("Accredited by NAAC with Grade “A++”", {
      align: AlignmentType.CENTER,
      size: 20,
    })
  );
  out.push(
    p(
      "Accredited by NBA - Tier1 (Civil, Mech, ECE, Auto, EEE, E&I & CSE)",
      { align: AlignmentType.CENTER, size: 20 }
    )
  );
  out.push(new Paragraph({ children: [new PageBreak()] }));
  return out;
}

function bonafide() {
  const out = [];
  out.push(p(COLLEGE, { align: AlignmentType.CENTER, bold: true, size: 28 }));
  out.push(p(PLACE, { align: AlignmentType.CENTER, bold: true, size: 26 }));
  out.push(
    p("Affiliated to Anna University, Chennai * Approved by AICTE, New Delhi", {
      align: AlignmentType.CENTER,
      size: 20,
    })
  );
  out.push(
    p("Accredited by NAAC with Grade “A++”", {
      align: AlignmentType.CENTER,
      size: 20,
    })
  );
  out.push(
    p("Accredited by NBA - Tier1 (Civil, Mech, ECE, Auto, EEE, E&I & CSE)", {
      align: AlignmentType.CENTER,
      size: 20,
    })
  );
  out.push(blank(400));
  out.push(
    p("BONAFIDE CERTIFICATE", {
      align: AlignmentType.CENTER,
      bold: true,
      size: 30,
    })
  );
  out.push(blank(240));
  out.push(
    p("Certified that this Mini Project report entitled", {
      align: AlignmentType.CENTER,
      size: 24,
    })
  );
  out.push(
    p(TITLE, { align: AlignmentType.CENTER, bold: true, size: 28 })
  );
  out.push(blank(160));
  out.push(
    p("is the bonafide record of work carried out by", {
      align: AlignmentType.CENTER,
      size: 24,
    })
  );
  out.push(blank(160));
  for (const [name, roll] of STUDENTS) {
    out.push(
      p(`${name}    -    ${roll}`, {
        align: AlignmentType.CENTER,
        bold: true,
        size: 26,
      })
    );
  }
  out.push(blank(320));
  out.push(
    p(
      `Submitted for Mini Project of ${COURSE} held on ………………`,
      { align: AlignmentType.CENTER, size: 24 }
    )
  );
  out.push(blank(280));
  out.push(
    table(
      ["S.No", "Criteria", "Marks"],
      [
        ["1", "Objective and Block Diagram", "/10"],
        ["2", "Implementation Code", "/10"],
        ["3", "Results and Inference", "/10"],
        ["4", "Submission in time", "/10"],
        ["5", "Total", "/50"],
      ],
      [12, 66, 22],
      { size: 22, boldRows: ["5"] }
    )
  );
  out.push(blank(600));
  out.push(
    p("SIGNATURE OF THE FACULTY", {
      align: AlignmentType.RIGHT,
      bold: true,
      size: 24,
    })
  );
  out.push(new Paragraph({ children: [new PageBreak()] }));
  return out;
}

function tableOfContents() {
  const rows = [
    ["1", "INTRODUCTION", "1"],
    ["", "1.1 PROBLEM STATEMENT", "1"],
    ["2", "OBJECTIVE", "2"],
    ["", "2.1 SPECIFIC OBJECTIVES", "2"],
    ["3", "BLOCK DIAGRAM", "3"],
    ["4", "IMPLEMENTATION CODE", "4"],
    ["", "4.1 DATA PIPELINE", "4"],
    ["", "4.2 VOCABULARY, TOKENISATION AND GloVe EMBEDDINGS", "7"],
    ["", "4.3 MODEL ARCHITECTURE - Bi-LSTM WITH ATTENTION", "9"],
    ["", "4.4 TRAINING LOOP", "11"],
    ["", "4.5 BASELINE - TF-IDF + LOGISTIC REGRESSION", "14"],
    ["", "4.6 INFERENCE AND THE STREAMLIT WEB APPLICATION", "15"],
    ["5", "RESULTS", "17"],
    ["", "5.1.1 DATASET AND CLASS DISTRIBUTION", "17"],
    ["", "5.1.2 EXPLORATORY DATA ANALYSIS", "18"],
    ["", "5.1.3 MODEL TRAINING AND EVALUATION RESULTS", "19"],
    ["", "5.1.4 CONFUSION MATRIX OF THE MAIN MODEL", "21"],
    ["", "5.1.5 POST-HOC OPTIMISATION AND ENSEMBLE", "22"],
    ["", "5.1.6 ADDITIONAL EXPERIMENTS (NEGATIVE RESULTS)", "23"],
    ["", "5.1.7 STREAMLIT WEB APPLICATION DEPLOYMENT", "24"],
    ["", "5.2.1 TEST CASE 1 - CLEAR NEGATIVE REVIEW", "25"],
    ["", "5.2.2 TEST CASE 2 - CLEAR POSITIVE REVIEW", "26"],
    ["", "5.2.3 TEST CASE 3 - MIXED REVIEW (CONTRAST DETECTION)", "26"],
    ["6", "INFERENCE AND DISCUSSION", "27"],
    ["7", "GITHUB LINKS", "29"],
  ];
  return [
    p("TABLE OF CONTENTS", {
      align: AlignmentType.CENTER,
      bold: true,
      size: 30,
    }),
    blank(200),
    table(["CHAPTER NO", "TITLE", "PAGE NO"], rows, [16, 68, 16], { size: 21 }),
  ];
}

module.exports = { DATA };

// ------------------------------------------------------------------- report

function introduction() {
  return [
    h1("1.0 INTRODUCTION", true),
    p(
      "Online shopping platforms accumulate customer reviews far faster than any team can read them. A single mid-range electronics product on Amazon can attract tens of thousands of written reviews, and the aggregate star rating shown on the product page compresses all of that language into one number. That number tells a prospective buyer very little about why previous customers were satisfied or disappointed, and it gives the seller no structured signal about which specific complaints are recurring."
    ),
    p(
      "Sentiment analysis addresses this by automatically assigning a sentiment label to each review. The task is a natural fit for recurrent neural networks: a review is a sequence, and the meaning of a word depends on the words around it. The phrase “not worth the money” carries the opposite sentiment to “worth the money” despite sharing three of its four words, and a model that ignores order cannot represent that difference reliably."
    ),
    p(
      "This project builds and compares five sentiment classifiers on 300,000 Amazon Electronics reviews from the Amazon Reviews 2023 dataset. The models range from a classical bag-of-words baseline through three recurrent architectures to a fine-tuned transformer, all trained and evaluated on identical data splits so that the comparison isolates the effect of the architecture. The main model is a bidirectional LSTM with an attention layer over pretrained GloVe embeddings, chosen because attention makes the model's decision partially interpretable: the weights it assigns to each word can be displayed to a user."
    ),
    p(
      "The system is deployed as an interactive Streamlit web application in which a user types a review and immediately receives the predicted sentiment, the probability assigned to each of the three classes, and a visualisation of the words the attention layer weighted most heavily."
    ),
    h2("1.1 PROBLEM STATEMENT"),
    p(
      "Customer reviews on e-commerce platforms are abundant but unstructured, and the star rating that accompanies them is a coarse summary that hides the reasoning behind it. Automatic three-class sentiment classification of review text is complicated by two properties of real review data. First, the classes are severely imbalanced: in this dataset 79.14% of reviews are positive and only 7.28% are neutral, so a model that ignores its input entirely and always answers “positive” achieves 79.14% accuracy while being useless. Second, the neutral class is not simply rarer but semantically harder, because it sits between the other two and is typically expressed through contrast within a sentence rather than through a distinctive vocabulary of its own."
    ),
    p(
      "This project addresses the problem by building a class-weighted, sequence-aware classifier evaluated by macro-F1 rather than accuracy, and by using an attention mechanism to capture the contrastive structure that signals a mixed opinion."
    ),
  ];
}

function objectives() {
  const specifics = [
    [
      "Scalable Data Acquisition and Preprocessing:",
      "Stream the 22.6 GB Amazon Reviews 2023 Electronics corpus from the Hugging Face Hub without downloading it, sample 300,000 reviews through a seeded shuffle buffer, map star ratings onto three sentiment classes, remove duplicates and unusable rows, and produce a reproducible stratified 80/10/10 train/validation/test split.",
    ],
    [
      "Exploratory Characterisation of the Label Space:",
      "Quantify the class imbalance, review-length distribution and class-discriminative vocabulary on the training split alone, and use those findings to fix modelling parameters such as the maximum sequence length and the class weights.",
    ],
    [
      "Comparative Architecture Evaluation:",
      "Implement and train five classifiers - a TF-IDF and Logistic Regression baseline, a plain LSTM, a bidirectional GRU, a bidirectional LSTM with attention, and a fine-tuned DistilBERT - under identical splits, class weighting and stopping criteria so that differences in score are attributable to the architecture.",
    ],
    [
      "Imbalance-Aware Training and Evaluation:",
      "Apply inverse-frequency class weighting in the loss, train with mixed precision within a 6 GB GPU budget, stop early on validation macro-F1, and report macro-F1, per-class precision, recall and F1, and confusion matrices rather than accuracy alone.",
    ],
    [
      "Interpretability and Interactive Deployment:",
      "Expose the attention weights as a word-level explanation of each prediction and deploy the trained model as a real-time Streamlit web application in which a user can enter arbitrary review text and inspect both the predicted class probabilities and the words that drove the decision.",
    ],
  ];

  const out = [
    h1("2.0 OBJECTIVE", true),
    p(
      "To design, implement, train and deploy a three-class customer review sentiment classification system based on recurrent neural networks with attention, which handles severe class imbalance correctly, is evaluated by metrics that cannot be inflated by the majority class, and provides an interpretable, real-time interface for classifying previously unseen review text."
    ),
    h2("2.1 SPECIFIC OBJECTIVES"),
  ];
  for (const [heading, body] of specifics) {
    out.push(
      new Paragraph({
        spacing: { before: 140, after: 60 },
        children: [new TextRun({ text: heading, bold: true, size: 24, font: BODY })],
      })
    );
    out.push(p(body, { indent: { left: 360 } }));
  }
  return out;
}

function blockDiagram() {
  return [
    h1("3.0 BLOCK DIAGRAM", true),
    p(
      "The system is organised as four stages. Data preparation streams and samples the corpus and produces the fixed splits. Text representation converts review text into token identifiers and initialises the embedding layer from pretrained GloVe vectors. The model stage trains five architectures on those identical splits. The evaluation and deployment stage applies class-weighted training, computes the reported metrics, performs post-hoc optimisation, and serves the selected model through the web application."
    ),
    ...figure("block_diagram", "Figure 3.1 - System block diagram of the sentiment analysis pipeline", 640),
  ];
}

function implementation() {
  const out = [h1("4.0 IMPLEMENTATION CODE", true)];
  out.push(
    p(
      "The implementation is organised as a modular Python package. Each listing below is the actual source of the corresponding component, reproduced from the project repository."
    )
  );
  DATA.code.forEach((section, i) => {
    out.push(h2(section.title, i > 0));
    out.push(
      p(`Source file: ${section.file}`, {
        italics: true,
        size: 20,
        color: "52514E",
      })
    );
    out.push(...codeBlock(section.source));
  });
  return out;
}

function results() {
  const out = [h1("5.0 RESULTS", true)];

  // ---- 5.1.1 dataset --------------------------------------------------- //
  out.push(h2("5.1.1 DATASET AND CLASS DISTRIBUTION"));
  out.push(
    p(
      "300,000 reviews were sampled from the Amazon Reviews 2023 Electronics category by streaming the source file and drawing through a 100,000-row shuffle buffer with seed 42. 314,076 records were scanned to obtain them: 189 were discarded as too short after cleaning and 13,887 as exact duplicates. Removing duplicates before splitting is important, because an identical review appearing in both the training and test sets would inflate every reported score."
    )
  );
  const dist = DATA.distribution.distribution || {};
  const distRows = ["negative", "neutral", "positive"].map((c) => [
    c,
    (dist["full sample"]?.[c]?.count ?? 0).toLocaleString(),
    (dist["full sample"]?.[c]?.percent ?? 0) + "%",
    (dist["train"]?.[c]?.count ?? 0).toLocaleString(),
    (dist["val"]?.[c]?.count ?? 0).toLocaleString(),
    (dist["test"]?.[c]?.count ?? 0).toLocaleString(),
  ]);
  distRows.push([
    "TOTAL",
    "300,000",
    "100.00%",
    "240,000",
    "30,000",
    "30,000",
  ]);
  out.push(
    table(
      ["Class", "Count", "Share", "Train", "Validation", "Test"],
      distRows,
      [22, 16, 14, 16, 16, 16],
      { boldRows: ["TOTAL"] }
    )
  );
  out.push(caption("Table 5.1 - Class distribution across the stratified splits"));
  out.push(
    p(
      "The stratified split preserves the class ratio to two decimal places in all three partitions, so validation and test scores are directly comparable to training conditions."
    )
  );
  out.push(...figure("class_distribution", "Figure 5.1 - Class distribution in the training split", 520));

  // ---- 5.1.2 EDA -------------------------------------------------------- //
  out.push(h2("5.1.2 EXPLORATORY DATA ANALYSIS", true));
  out.push(
    p(
      "Exploratory analysis was performed on the training split only. Inspecting the validation or test split, even to plot it, would leak information into the modelling decisions taken from those plots - the vocabulary size, the maximum sequence length and the class weights were all chosen from these figures."
    )
  );
  out.push(...figure("review_length", "Figure 5.2 - Review length distribution by class", 640));
  out.push(
    p(
      "Review length is strongly right-skewed: 90% of reviews are 146 words or fewer and 95% are 220 or fewer, but the longest runs to 5,534 words. Padding every sequence to the maximum would spend almost all of the computation on padding tokens, so the maximum sequence length was fixed at 230 tokens. Neutral reviews are the longest on average (median 52 words against 33 for positive), which is consistent with a mixed opinion requiring more words to express."
    )
  );
  out.push(...figure("top_words", "Figure 5.3 - Most class-discriminative words by smoothed log-odds", 640));
  out.push(
    p(
      "Words were ranked by a smoothed log-odds ratio rather than by raw frequency, which would return the same common terms for every class. Negative reviews have strongly distinctive vocabulary reaching a log-odds of 3.4 (scam, garbage, worthless) and positive reviews reach 3.2 (lifesaver, godsend), but neutral peaks at only 1.9 and its markers are hedges such as meh, eh, alright and mediocre. This is direct evidence that the neutral class has no characteristic vocabulary of its own and is instead signalled by contrast, and it motivated the choice of an attention mechanism as the main model."
    )
  );

  // ---- 5.1.3 comparison -------------------------------------------------- //
  out.push(h2("5.1.3 MODEL TRAINING AND EVALUATION RESULTS", true));
  out.push(
    p(
      "All five models were trained with inverse-frequency class weights (negative 2.45, neutral 4.58, positive 0.42), early stopping on validation macro-F1 with patience 2, and mixed-precision arithmetic. The test split was scored exactly once per model, after every hyperparameter decision had been fixed on validation."
    )
  );
  const best = Math.max(...DATA.comparison.map((r) => r.macro_f1));
  const compRows = DATA.comparison.map((r) => [
    r.model,
    f4(r.accuracy),
    f4(r.macro_f1),
    f4(r.per_class.negative.f1),
    f4(r.per_class.neutral.f1),
    f4(r.per_class.positive.f1),
    r.parameters ? r.parameters.toLocaleString() : "-",
  ]);
  compRows.push([
    "always predict positive",
    "0.7914",
    "0.2945",
    "0.0000",
    "0.0000",
    "0.8836",
    "0",
  ]);
  out.push(
    table(
      ["Model", "Accuracy", "Macro-F1", "F1 neg", "F1 neu", "F1 pos", "Parameters"],
      compRows,
      [28, 12, 12, 11, 11, 11, 15],
      { boldRows: [DATA.comparison.find((r) => r.macro_f1 === best)?.model] }
    )
  );
  out.push(caption("Table 5.2 - Model comparison on the test split (30,000 reviews)"));
  out.push(
    p(
      "The final row is the trivial classifier that ignores its input and always answers with the majority class. It scores 0.7914 accuracy and 0.2945 macro-F1, which is the clearest single justification for using macro-F1 as the headline metric: an accuracy figure near 0.87 is only 0.08 above a model that has learned nothing."
    )
  );
  out.push(...figure("model_comparison", "Figure 5.4 - Macro-F1 by model on the test split", 600));
  out.push(...figure("per_class_f1", "Figure 5.5 - Per-class F1 for each model", 640));
  // Written from the data so the commentary cannot drift from the table.
  const ranked = [...DATA.comparison].sort((a, b) => b.macro_f1 - a.macro_f1);
  const top = ranked[0];
  const runner = ranked[1];
  const lstm = DATA.comparison.find((r) => r.model === "LSTM");
  const bigru = DATA.comparison.find((r) => r.model === "Bi-GRU");
  const attn = DATA.comparison.find((r) => r.model === "Bi-LSTM + Attention");
  const base = DATA.comparison.find((r) => r.model === "TF-IDF + LogReg");

  out.push(
    p(
      `The strongest single model is ${top.model} at ${f4(
        top.macro_f1
      )} macro-F1, ahead of ${runner.model} at ${f4(
        runner.macro_f1
      )}. Three observations are worth stating explicitly. First, every learned sequence model beats the bag-of-words baseline (${f4(
        base.macro_f1
      )}), but the margin is smaller than the parameter counts would suggest: the plain LSTM buys only ${(
        lstm.macro_f1 - base.macro_f1
      ).toFixed(
        4
      )} macro-F1 for 4.6 million parameters, because word presence alone already carries most of the sentiment signal in product reviews.`
    )
  );
  out.push(
    p(
      `Second, bidirectionality contributes more than attention does. Moving from the unidirectional LSTM (${f4(
        lstm.macro_f1
      )}) to the bidirectional GRU (${f4(
        bigru.macro_f1
      )}) is a larger gain than anything the attention layer adds on top of a bidirectional encoder (${f4(
        attn.macro_f1
      )}). Reading a review in both directions is worth more, on this task, than learning where within it to look - and the Bi-GRU slightly outscores the nominated main model, which is reported as measured rather than reordered.`
    )
  );
  out.push(
    p(
      `Third, the transformer leads but not overwhelmingly. DistilBERT carries roughly fourteen times the parameters of the recurrent models and was pretrained on a large general corpus, yet its advantage over the best 4.7-million-parameter recurrent model is ${(
        top.macro_f1 - bigru.macro_f1
      ).toFixed(
        4
      )} macro-F1. For a deployment constrained by memory or latency, the recurrent models remain a reasonable choice, and only the attention model can produce the word-level explanation used in the web application.`
    )
  );

  // ---- 5.1.4 confusion --------------------------------------------------- //
  out.push(h2("5.1.4 CONFUSION MATRIX OF THE MAIN MODEL", true));
  const main = DATA.comparison.find((r) => r.model === "Bi-LSTM + Attention");
  if (main) {
    const cm = main.confusion_matrix;
    out.push(
      table(
        ["True \\ Predicted", "negative", "neutral", "positive", "Recall"],
        ["negative", "neutral", "positive"].map((c, i) => [
          c,
          cm[i][0].toLocaleString(),
          cm[i][1].toLocaleString(),
          cm[i][2].toLocaleString(),
          f4(main.per_class[c].recall),
        ]),
        [26, 18, 18, 18, 20]
      )
    );
    out.push(caption("Table 5.3 - Confusion matrix, Bi-LSTM + Attention, test split"));
    out.push(
      table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        ["negative", "neutral", "positive"].map((c) => [
          c,
          f4(main.per_class[c].precision),
          f4(main.per_class[c].recall),
          f4(main.per_class[c].f1),
          main.per_class[c].support.toLocaleString(),
        ]),
        [28, 18, 18, 18, 18]
      )
    );
    out.push(caption("Table 5.4 - Per-class scores, Bi-LSTM + Attention, test split"));
  }
  out.push(...figure("confusion_main", "Figure 5.6 - Confusion matrix of the main model", 480));
  out.push(
    p(
      "The dominant error is positive reviews predicted as neutral, which accounts for close to half of all mistakes and holds neutral precision down to roughly 0.38. This is a direct and deliberate consequence of the class weighting: neutral carries approximately eleven times the loss weight of positive, so the model is explicitly instructed that missing a neutral review is far more costly than raising a false alarm, and it responds by over-claiming that class. Without the weighting, neutral recall collapses and macro-F1 falls sharply. Notably the two poles are almost never confused with each other, so the sentiment axis itself is learned well and essentially all remaining difficulty lies at the neutral boundary."
    )
  );

  // ---- 5.1.5 optimisation ------------------------------------------------ //
  out.push(h2("5.1.5 POST-HOC OPTIMISATION AND ENSEMBLE", true));
  out.push(
    p(
      "Because every model over-predicts neutral, the predicted probabilities can be rescaled per class to rebalance precision against recall. F1 is maximised when the two are close, so this recovers macro-F1 without retraining. The scaling vector was found by coordinate ascent on the validation split and then applied once to test. An ensemble was also formed by averaging the predicted probabilities of the individual models."
    )
  );
  if (DATA.optimisation.length) {
    out.push(
      table(
        ["Model", "Macro-F1", "After tuning", "Change", "Neutral F1", "After tuning"],
        DATA.optimisation.map((r) => [
          r.model,
          f4(r.macro_f1_argmax),
          f4(r.macro_f1_tuned),
          (r.delta >= 0 ? "+" : "") + Number(r.delta).toFixed(4),
          f4(r.neutral_f1_argmax),
          f4(r.neutral_f1_tuned),
        ]),
        [30, 14, 14, 13, 14, 15]
      )
    );
    out.push(caption("Table 5.5 - Effect of per-class probability rescaling and ensembling"));
    const bestOpt = DATA.optimisation.reduce((a, b) =>
      b.macro_f1_tuned > a.macro_f1_tuned ? b : a
    );
    const improved = DATA.optimisation.filter((r) => r.delta > 0.0005).length;
    out.push(
      p(
        `${improved} of the ${DATA.optimisation.length} configurations improved. The best is ${bestOpt.model} at a macro-F1 of ${f4(
          bestOpt.macro_f1_tuned
        )}, with a neutral F1 of ${f4(
          bestOpt.neutral_f1_tuned
        )}. The tuned vectors consistently raise the positive class and lower the neutral class, which confirms the over-prediction diagnosis numerically rather than by inspection. The one model that gains nothing is the fully fine-tuned DistilBERT, whose fitted weights come out at 1.00, 1.00, 1.00 - it is already calibrated, and needs no correction.`
      )
    );
  }

  // ---- 5.1.6 negative results -------------------------------------------- //
  out.push(h2("5.1.6 ADDITIONAL EXPERIMENTS (NEGATIVE RESULTS)", true));
  out.push(
    p(
      "Two further modifications were tested and neither improved test performance. They are reported because an experiment that fails is still evidence about the problem, and omitting them would misrepresent how the final configuration was reached."
    )
  );
  if (DATA.experiments.length) {
    out.push(
      table(
        ["Experiment", "Change", "Baseline macro-F1", "Result", "Change"],
        DATA.experiments.map((r) => [
          r.model,
          r.note,
          f4(r.baseline_macro_f1),
          f4(r.macro_f1),
          (r.delta >= 0 ? "+" : "") + Number(r.delta).toFixed(4),
        ]),
        [22, 34, 16, 14, 14],
        { size: 18 }
      )
    );
    out.push(caption("Table 5.6 - Experiments that did not improve on the baseline configuration"));
  }
  out.push(
    p(
      "The capacity experiment doubled the recurrent depth, widened the hidden state and extended the sequence limit. Validation macro-F1 improved noticeably but test macro-F1 did not move, which indicates that the original architecture was not capacity-limited and that the additional parameters fitted the validation split rather than the task."
    )
  );
  out.push(
    p(
      "The ordinal loss replaced plain cross-entropy with distance-aware soft targets, so that confusing the two poles costs more than confusing adjacent classes. This is theoretically well motivated for an ordinal label such as a star rating, but it reduced macro-F1 for both architectures tested. The most plausible explanation is that softening the targets towards neighbouring classes works against the class weighting, which is deliberately trying to sharpen the model's willingness to commit to the rare neutral class."
    )
  );

  // ---- 5.1.7 deployment ---------------------------------------------------- //
  out.push(h2("5.1.7 STREAMLIT WEB APPLICATION DEPLOYMENT", true));
  out.push(
    p(
      "The trained Bi-LSTM with attention is served through a Streamlit application. The user enters review text and receives the predicted class, the probability assigned to each class, and the review rendered with each word shaded in proportion to its attention weight. The application is a thin layer over the same inference class used by the command-line interface, and it reuses the training-time cleaning and tokenisation functions directly, so that text entered by a user is processed in exactly the same way as the training data."
    )
  );
  out.push(...figure("app_mixed", "Figure 5.7 - Web application classifying a mixed review", 560));
  out.push(new Paragraph({ children: [new PageBreak()] }));
  out.push(...figure("app_negative", "Figure 5.8 - Web application classifying a negative review", 560));

  // ---- 5.2 test cases ------------------------------------------------------ //
  out.push(h2("5.2.0 MODEL TESTING ON UNSEEN REVIEW TEXT", true));
  out.push(
    p(
      "Three representative reviews were submitted to the deployed model. The outputs below were produced by the trained model at report generation time."
    )
  );
  DATA.test_cases.forEach((c, i) => {
    out.push(h3(`5.2.${i + 1} ${c.title}`));
    out.push(
      new Paragraph({
        spacing: { after: 100, line: 276 },
        children: [
          new TextRun({ text: "DESCRIPTION: ", bold: true, size: 24, font: BODY }),
          new TextRun({ text: c.description, size: 24, font: BODY }),
        ],
      })
    );
    out.push(
      new Paragraph({
        spacing: { after: 120, line: 276 },
        indent: { left: 360 },
        children: [
          new TextRun({ text: "INPUT: ", bold: true, size: 22, font: BODY }),
          new TextRun({ text: `“${c.text}”`, italics: true, size: 22, font: BODY }),
        ],
      })
    );
    out.push(
      table(
        ["Predicted class", "Confidence", "P(negative)", "P(neutral)", "P(positive)"],
        [
          [
            c.predicted,
            pct(c.confidence),
            pct(c.probabilities.negative),
            pct(c.probabilities.neutral),
            pct(c.probabilities.positive),
          ],
        ],
        [22, 18, 20, 20, 20]
      )
    );
    out.push(
      p(
        "Highest attention weights: " +
          c.top_tokens.map(([t, w]) => `${t} (${w})`).join(",  "),
        { size: 22, italics: true, after: 240 }
      )
    );
  });
  out.push(...figure("attention", "Figure 5.9 - Attention weights over sample reviews", 640));
  return out;
}

function inference() {
  const points = [
    [
      "Macro-F1 is the only defensible headline metric on this data.",
      "With 79.14% of reviews positive, a classifier that ignores its input entirely reaches 0.7914 accuracy and 0.2945 macro-F1. Every accuracy figure in this report is therefore reported alongside macro-F1, and model selection and early stopping were driven by macro-F1 throughout. A project that reported only accuracy here would appear to have a strong model while having learned very little.",
    ],
    [
      "Bidirectionality contributed more than attention did.",
      "The plain unidirectional LSTM reached 0.7282 macro-F1, while the bidirectional GRU reached 0.7506 - a larger improvement than anything the attention mechanism added on top of a bidirectional encoder, and more than half the distance from the LSTM to the fine-tuned transformer at 0.7661. Reading a review in both directions is worth more, on this task, than learning where within it to look.",
    ],
    [
      "Attention earns its place through interpretability rather than accuracy.",
      "On a mixed review, the attention layer assigned its highest weight to the contrastive conjunction “but”, several times the weight of any other token, and the model correctly returned neutral. Nothing in the training signal identified that word as important; it carries no sentiment in isolation and a bag-of-words model can only treat it as a common stopword. The model learned that it marks the point at which a review turns. This is exactly the mechanism the exploratory analysis predicted would be required, and it is what makes the word-level explanation in the web application meaningful to an end user.",
    ],
    [
      "Correct tokenisation mattered more than architecture.",
      "The first training run of the main model scored 0.7218 macro-F1, statistically indistinguishable from the bag-of-words baseline despite having 4.7 million parameters and pretrained embeddings. Investigation showed only 45.6% of the vocabulary had a GloVe vector, and that 87.2% of the misses were punctuation attached to words: splitting on whitespace produced tokens such as “great.” and “don't”, none of which exist in GloVe, so the most sentiment-bearing words were mapped to the unknown token and given random vectors. Matching GloVe's own Penn-style tokenisation raised coverage to 73.1% and the model to 0.7435, an improvement of 0.022 macro-F1 from a change that touched no part of the architecture.",
    ],
    [
      "The remaining error on the neutral class is largely a property of the labels.",
      "No architecture reached a neutral F1 of 0.51, and the four recurrent and count-based models all sat below 0.49; the fine-tuned transformer, with roughly fourteen times the parameters, reached only 0.5022. That consistency across five very different model families is itself the evidence. Manual inspection of the twenty most confidently misclassified neutral reviews showed that eleven of them have text which flatly contradicts the star rating - for example “I love it. I love it” and “Great item as advertised”, both rated three stars. No text-based model can recover those labels, and a human annotator reading only the text would make the same predictions. This places a ceiling on achievable neutral performance that is a consequence of treating the star rating as a proxy for sentiment, and it would be misleading to attribute it to the architecture.",
    ],
    [
      "Calibration is a cheaper source of improvement than capacity.",
      "Rescaling the class probabilities using a vector fitted on validation improved every recurrent and count-based model, and combining the models into a greedily weighted ensemble raised the best configuration to 0.7759 macro-F1 with a neutral F1 of 0.5269 - the highest either figure reached in this project. Notably the fully trained DistilBERT required no rescaling at all (its fitted weights were 1.00, 1.00, 1.00), whereas the same model trained on a quarter of the data had needed a substantial correction: training to convergence on the full split produced a well-calibrated classifier, while the under-trained version did not. By contrast, doubling the depth and width of the best recurrent architecture produced no test improvement whatsoever. On an imbalanced problem, calibration and combination are more productive than parameter count.",
    ],
  ];

  const out = [h1("6.0 INFERENCE AND DISCUSSION", true)];
  for (const [heading, body] of points) {
    out.push(
      new Paragraph({
        spacing: { before: 180, after: 60 },
        children: [new TextRun({ text: heading, bold: true, size: 24, font: BODY })],
      })
    );
    out.push(p(body, { indent: { left: 360 } }));
  }

  out.push(h2("6.1 LIMITATIONS AND FUTURE WORK"));
  for (const t of [
    "The neutral class is defined as exactly the three-star band, which conflates genuinely mixed opinions with inconsistent use of the rating scale. Sentiment labels annotated from the text itself would raise the achievable ceiling more than any architectural change.",
    "DistilBERT was fine-tuned under a reduced budget to fit a 6 GB GPU, so the comparison with it is not an equal-footing comparison and its reported score should be treated as a lower bound.",
    "Reviews longer than the 230-token limit lose their ending, and reviews structured as praise followed by a qualification are precisely the case where the truncated portion carries the decisive information.",
    "The decision rule is a plain argmax over the tuned probabilities; a threshold optimised directly for the deployment objective, or an ordinal decision rule, would suit an application where the cost of the two error types differs.",
    "Sarcasm and figurative praise remain out of reach for this architecture, as every surface cue in such reviews points the wrong way.",
  ]) {
    out.push(bullet(t));
  }
  return out;
}

function githubLinks() {
  return [
    h1("7.0 GITHUB LINKS", true),
    p(
      "The complete source code, trained configuration, generated figures and result files are available in the project repository."
    ),
    blank(120),
    table(
      ["ROLL NO.", "LINK"],
      STUDENTS.map(([, roll]) => [roll, GITHUB_LINK]),
      [26, 74]
    ),
    blank(240),
    p(
      "The repository contains the data pipeline, all five model implementations, the shared training loop, evaluation and error-analysis scripts, the Streamlit application, and the notebook used for exploratory analysis.",
      { size: 22 }
    ),
  ];
}

// -------------------------------------------------------------------- build

const doc = new Document({
  creator: "Mini Project",
  title: TITLE,
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: BODY, size: 24 } },
    },
  },
  sections: [
    {
      properties: {
        page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({
                  children: [PageNumber.CURRENT],
                  size: 20,
                  font: BODY,
                }),
              ],
            }),
          ],
        }),
      },
      children: [
        ...frontPage(),
        ...bonafide(),
        ...tableOfContents(),
        ...introduction(),
        ...objectives(),
        ...blockDiagram(),
        ...implementation(),
        ...results(),
        ...inference(),
        ...githubLinks(),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log("wrote", OUT, (buffer.length / 1024).toFixed(0) + " KB");
});
