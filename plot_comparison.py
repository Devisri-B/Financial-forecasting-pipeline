import matplotlib.pyplot as plt
import numpy as np

# Performance comparison
models = ['Single-Ticker\nLSTM\n(570 samples)', 'Multi-Ticker\nLSTM\n(7,666 samples)', 'Multi-Ticker\nEnsemble\n(7,666 samples)']
r2_scores = [0.111, 0.9826, 0.9986]
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
bars = ax1.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (R²=1.0)')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{score:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Sample size impact
sample_sizes = [570, 7666]
lstm_r2 = [0.111, 0.9826]

ax2.plot(sample_sizes, lstm_r2, marker='o', markersize=12, linewidth=3, color='#2ca02c', label='LSTM R²')
ax2.set_xlabel('Training Samples', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('Impact of Training Data Size', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# Annotate points
ax2.annotate(f'{lstm_r2[0]:.3f}', xy=(sample_sizes[0], lstm_r2[0]), 
             xytext=(sample_sizes[0]*0.7, lstm_r2[0]-0.1),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax2.annotate(f'{lstm_r2[1]:.4f}', xy=(sample_sizes[1], lstm_r2[1]), 
             xytext=(sample_sizes[1]*1.1, lstm_r2[1]-0.05),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('outputs/performance_comparison.png', dpi=150, bbox_inches='tight')
print(' Saved performance comparison to outputs/performance_comparison.png')
print(f'   Single-ticker LSTM: R²={r2_scores[0]:.3f}')
print(f'   Multi-ticker LSTM: R²={r2_scores[1]:.4f} (88x improvement)')
print(f'   Multi-ticker Ensemble: R²={r2_scores[2]:.4f} (+1.6% over LSTM)')
