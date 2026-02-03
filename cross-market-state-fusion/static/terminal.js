// RL Training Terminal V2.0 Logic

const UPDATE_INTERVAL = 3000; // 3 seconds

// Chart Configs
const commonLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#8b949e', family: 'JetBrains Mono' },
    margin: { t: 10, b: 30, l: 40, r: 10 },
    xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
    yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
    showlegend: true,
    legend: { x: 0, y: 1, orientation: 'h' }
};

function initCharts() {
    Plotly.newPlot('loss-chart', [
        { name: 'Entropy', y: [], mode: 'lines', line: { color: '#58a6ff' } },
        { name: 'Policy Loss', y: [], mode: 'lines', line: { color: '#da3633' }, yaxis: 'y2' }
    ], {
        ...commonLayout,
        yaxis: { ...commonLayout.yaxis, title: 'Entropy' },
        yaxis2: { overlaying: 'y', side: 'right', title: 'Loss' }
    });

    Plotly.newPlot('pnl-chart', [
        { name: 'PnL', y: [], fill: 'tozeroy', type: 'scatter', line: { color: '#3fb950' } }
    ], {
        ...commonLayout,
        yaxis: { ...commonLayout.yaxis, title: 'USD' }
    });
}

function updatePerformance() {
    $.getJSON('/api/performance', function(data) {
        if ($.isEmptyObject(data)) return;

        // Scorecard
        $('#net-pnl').text((data.net_pnl > 0 ? '+' : '') + data.net_pnl);
        $('#net-pnl').removeClass('positive negative').addClass(data.net_pnl >= 0 ? 'positive' : 'negative');
        
        $('#sharpe-ratio').text(data.sharpe);
        $('#profit-factor').text(data.profit_factor);
        $('#max-drawdown').text(data.max_drawdown);

        // Health
        $('#health-score').text(data.health_score);
        $('#health-bar').css('width', data.health_score + '%');
        $('#health-status-text').text(data.health_status);
        
        // Live Status
        $('#total-trades').text(data.total_trades);
    });
}

function updateBehavior() {
    $.getJSON('/api/behavior', function(data) {
        if ($.isEmptyObject(data)) return;

        $('#avg-holding').text(data.avg_holding_time + 's');
        $('#long-short-bar').css('width', data.long_percentage + '%');
        $('#long-pct').text(data.long_percentage);
        $('#short-pct').text(data.short_percentage);
    });
}

function updateUpdates() {
    $.getJSON('/api/updates', function(data) {
        if (data.data && data.data.length === 0) return;

        $('#total-updates').text(data.update_nums.length);
        
        const lastEntropy = data.entropy[data.entropy.length - 1];
        $('#current-entropy').text(lastEntropy ? lastEntropy.toFixed(3) : '--');

        // Update Loss/Entropy Chart
        Plotly.update('loss-chart', {
            y: [data.entropy, data.policy_loss]
        });

        // Update PnL Chart
        Plotly.update('pnl-chart', {
            y: [data.cumulative_pnl]
        });
        
        const lastPnL = data.cumulative_pnl[data.cumulative_pnl.length - 1];
         $('#pnl-header-val').text(lastPnL ? '$' + lastPnL.toFixed(2) : '--');
    });
}

function updateLogs() {
    $.getJSON('/api/trades/recent', function(data) {
        const tbody = $('#trades-table-body');
        tbody.empty();
        data.reverse().slice(0, 20).forEach(trade => { // Show last 20
            const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
            const time = new Date(trade.timestamp).toLocaleTimeString();
            const row = `
                <tr style="border-bottom: 1px solid #21262d;">
                    <td style="padding: 5px;">${time}</td>
                    <td style="color: #58a6ff;">${trade.asset}</td>
                    <td class="${pnlClass}">${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}</td>
                </tr>
            `;
            tbody.append(row);
        });
    });
}

$(document).ready(function() {
    initCharts();
    
    // Initial fetch
    updatePerformance();
    updateBehavior();
    updateUpdates();
    updateLogs();

    // Loop
    setInterval(() => {
        updatePerformance();
        updateBehavior();
        updateUpdates();
        updateLogs();
        $('#last-update').text('Last update: ' + new Date().toLocaleTimeString());
    }, UPDATE_INTERVAL);
});
