/* === Chart.js dark theme defaults === */
Chart.defaults.color = '#8b8fa3';
Chart.defaults.borderColor = '#2a2d3a';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 12;

const AMBER = '#e8a838';
const GREEN = '#4ade80';
const BLUE = '#60a5fa';
const RED = '#f87171';

/* === F1 Comparison hero chart === */
new Chart(document.getElementById('f1ComparisonChart'), {
  type: 'bar',
  data: {
    labels: [
      'Gemma 2B Full LoRA',
      'Llama 3.2 3B LoRA',
      'Mistral 7B CF LoRA',
      'Gemma 2B CF LoRA',
      'Llama 1B LoRA',
      'Zero-shot Llama 3B',
      'Zero-shot Mistral 7B',
      'Zero-shot Gemma 2B'
    ],
    datasets: [{
      data: [0.916, 0.856, 0.760, 0.249, 0.196, 0.108, 0.095, 0.063],
      backgroundColor: [
        AMBER, AMBER, AMBER, AMBER, AMBER,
        '#555', '#555', '#555'
      ],
      borderRadius: 4,
      barThickness: 24
    }]
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1c1f2a',
        borderColor: '#2a2d3a',
        borderWidth: 1,
        titleColor: '#e2e4ea',
        bodyColor: '#e2e4ea',
        callbacks: {
          label: (ctx) => `Macro-F1: ${ctx.raw.toFixed(3)}`
        }
      }
    },
    scales: {
      x: { min: 0, max: 1.0, grid: { color: '#2a2d3a' }, ticks: { callback: (v) => v.toFixed(1) } },
      y: { grid: { display: false } }
    }
  }
});

/* === Per-Class F1 multi-model bar chart === */
new Chart(document.getElementById('perClassChart'), {
  type: 'bar',
  data: {
    labels: ['validation_error', 'distraction', 'comparison_shopping', 'accidental_exit', 'bot', 'committed_leave'],
    datasets: [
      {
        label: 'Gemma 2B Full',
        data: [0.957, 1.0, 0.9, 1.0, 0.889, 0.75],
        backgroundColor: AMBER,
        borderRadius: 4
      },
      {
        label: 'Llama 3B',
        data: [1.0, 0.947, 0.588, 1.0, 1.0, 0.6],
        backgroundColor: GREEN,
        borderRadius: 4
      },
      {
        label: 'Mistral 7B CF',
        data: [0.923, 0.75, 0.636, 0.875, 1.0, 0.375],
        backgroundColor: BLUE,
        borderRadius: 4
      }
    ]
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      legend: {
        labels: { usePointStyle: true, pointStyle: 'rectRounded', padding: 16 }
      },
      tooltip: {
        backgroundColor: '#1c1f2a',
        borderColor: '#2a2d3a',
        borderWidth: 1,
        titleColor: '#e2e4ea',
        bodyColor: '#e2e4ea',
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}`
        }
      }
    },
    scales: {
      x: { min: 0, max: 1.0, grid: { color: '#2a2d3a' }, ticks: { callback: (v) => v.toFixed(1) } },
      y: { grid: { display: false } }
    }
  }
});

/* === ECE comparison grouped bar chart === */
new Chart(document.getElementById('eceChart'), {
  type: 'bar',
  data: {
    labels: ['Gemma 2B', 'Llama 3B', 'Mistral 7B'],
    datasets: [
      { label: 'Verbalized', data: [0.145, 0.164, 0.245], backgroundColor: AMBER, borderRadius: 4 },
      { label: 'Logprob Raw', data: [0.103, 0.098, 0.071], backgroundColor: GREEN, borderRadius: 4 },
      { label: 'Calibrated', data: [0.056, 0.094, 0.075], backgroundColor: BLUE, borderRadius: 4 }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        labels: { usePointStyle: true, pointStyle: 'rectRounded', padding: 16 }
      },
      tooltip: {
        backgroundColor: '#1c1f2a',
        borderColor: '#2a2d3a',
        borderWidth: 1,
        titleColor: '#e2e4ea',
        bodyColor: '#e2e4ea',
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}`
        }
      }
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        grid: { color: '#2a2d3a' },
        title: { display: true, text: 'ECE', color: '#8b8fa3' },
        beginAtZero: true
      }
    }
  }
});

/* === Reliability diagram === */
new Chart(document.getElementById('reliabilityChart'), {
  type: 'line',
  data: {
    labels: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    datasets: [
      {
        label: 'Perfect Calibration',
        data: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        borderColor: '#555',
        borderDash: [6, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      },
      {
        label: 'Logprob Raw',
        data: [null, null, null, null, 0.0, 0.6667, 0.5, 0.75, 1.0, 1.0],
        borderColor: AMBER,
        backgroundColor: AMBER,
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        spanGaps: false,
        fill: false
      },
      {
        label: 'Calibrated',
        data: [null, null, null, null, null, 0.0, 0.6667, 0.6667, 0.8, 0.9744],
        borderColor: GREEN,
        backgroundColor: GREEN,
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        spanGaps: false,
        fill: false
      }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        labels: { usePointStyle: true, padding: 16 }
      },
      tooltip: {
        backgroundColor: '#1c1f2a',
        borderColor: '#2a2d3a',
        borderWidth: 1,
        titleColor: '#e2e4ea',
        bodyColor: '#e2e4ea'
      }
    },
    scales: {
      x: {
        type: 'linear',
        min: 0,
        max: 1,
        grid: { color: '#2a2d3a' },
        title: { display: true, text: 'Confidence', color: '#8b8fa3' }
      },
      y: {
        min: 0,
        max: 1,
        grid: { color: '#2a2d3a' },
        title: { display: true, text: 'Accuracy', color: '#8b8fa3' }
      }
    }
  }
});

/* === Confusion matrix (HTML table) === */
(function () {
  const CM_CLASSES = ['val_err', 'distract', 'comp_shop', 'acc_exit', 'bot', 'commit_leave'];
  const CM_MATRIX = [
    [10, 1, 0, 0, 0, 1],
    [0, 10, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 3],
    [0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 4, 0],
    [0, 0, 1, 0, 0, 7]
  ];

  const container = document.getElementById('confusionContainer');
  const table = document.createElement('table');
  table.className = 'cm-table';

  // Header row
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  headerRow.appendChild(document.createElement('th')); // empty corner
  CM_CLASSES.forEach((cls) => {
    const th = document.createElement('th');
    th.textContent = cls;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body rows
  const tbody = document.createElement('tbody');
  CM_MATRIX.forEach((row, i) => {
    const tr = document.createElement('tr');
    const rowHeader = document.createElement('th');
    rowHeader.textContent = CM_CLASSES[i];
    tr.appendChild(rowHeader);

    row.forEach((val, j) => {
      const td = document.createElement('td');
      td.textContent = val;
      if (val > 0) {
        if (i === j) {
          // Diagonal: amber
          td.style.backgroundColor = `rgba(232, 168, 56, ${(val / 12).toFixed(2)})`;
        } else {
          // Off-diagonal: red
          td.style.backgroundColor = `rgba(248, 113, 113, ${(val / 3).toFixed(2)})`;
        }
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
})();

/* === Training loss curves === */
new Chart(document.getElementById('lossChart'), {
  type: 'line',
  data: {
    labels: ['Epoch 1', 'Epoch 2', 'Epoch 3'],
    datasets: [
      {
        label: 'Llama 3B',
        data: [0.459, 0.361, 0.354],
        borderColor: AMBER,
        backgroundColor: AMBER,
        borderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: false,
        tension: 0.3
      },
      {
        label: 'Gemma 2B Full',
        data: [0.488, 0.399, 0.391],
        borderColor: GREEN,
        backgroundColor: GREEN,
        borderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: false,
        tension: 0.3
      },
      {
        label: 'Gemma 2B CF',
        data: [1.199, 0.857, 0.829],
        borderColor: RED,
        backgroundColor: RED,
        borderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: false,
        tension: 0.3
      },
      {
        label: 'Mistral 7B',
        data: [0.477, 0.399, 0.391],
        borderColor: BLUE,
        backgroundColor: BLUE,
        borderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: false,
        tension: 0.3
      }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        labels: { usePointStyle: true, padding: 16 }
      },
      tooltip: {
        backgroundColor: '#1c1f2a',
        borderColor: '#2a2d3a',
        borderWidth: 1,
        titleColor: '#e2e4ea',
        bodyColor: '#e2e4ea',
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}`
        }
      }
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        grid: { color: '#2a2d3a' },
        title: { display: true, text: 'Validation Loss', color: '#8b8fa3' }
      }
    }
  }
});
