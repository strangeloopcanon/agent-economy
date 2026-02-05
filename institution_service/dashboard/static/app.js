/**
 * Institution Service Dashboard - Real-time Visualization
 */

class Dashboard {
    constructor() {
        this.state = null;
        this.events = [];
        this.eventSource = null;
        this.init();
    }

    init() {
        this.connectEventStream();
        this.fetchInitialData();
    }

    async fetchInitialData() {
        try {
            // Fetch initial state
            const stateRes = await fetch('/api/state');
            if (stateRes.ok) {
                this.state = await stateRes.json();
                this.renderState();
            }

            // Fetch initial events
            const eventsRes = await fetch('/api/events');
            if (eventsRes.ok) {
                this.events = await eventsRes.json();
                this.renderEvents();
            }

            this.updateConnectionStatus('connected');
        } catch (err) {
            console.error('Failed to fetch initial data:', err);
            this.updateConnectionStatus('error');
        }
    }

    connectEventStream() {
        this.eventSource = new EventSource('/api/events/stream');

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'state') {
                    this.state = data.data;
                    this.renderState();
                } else {
                    this.events.push(data);
                    // Keep only last 100 events
                    if (this.events.length > 100) {
                        this.events = this.events.slice(-100);
                    }
                    this.renderEvents();
                }
            } catch (err) {
                console.error('Failed to parse event:', err);
            }
        };

        this.eventSource.onopen = () => {
            this.updateConnectionStatus('connected');
        };

        this.eventSource.onerror = () => {
            this.updateConnectionStatus('error');
            // Reconnect after 5 seconds
            setTimeout(() => {
                if (this.eventSource.readyState === EventSource.CLOSED) {
                    this.connectEventStream();
                }
            }, 5000);
        };
    }

    updateConnectionStatus(status) {
        const el = document.getElementById('connection-status');
        el.className = 'status ' + status;
        switch (status) {
            case 'connected':
                el.textContent = '● Live';
                break;
            case 'error':
                el.textContent = '● Disconnected';
                break;
            default:
                el.textContent = '● Connecting...';
        }
    }

    renderState() {
        if (!this.state) return;

        // Update header
        document.getElementById('run-id').textContent = `Run: ${this.state.run_id}`;
        document.getElementById('round-id').textContent = `Round: ${this.state.round_id}`;

        // Render tasks
        this.renderTasks();

        // Render workers
        this.renderWorkers();
    }

    renderTasks() {
        const columns = {
            'TODO': document.getElementById('todo-tasks'),
            'ASSIGNED': document.getElementById('assigned-tasks'),
            'REVIEW': document.getElementById('review-tasks'),
            'DONE': document.getElementById('done-tasks'),
        };

        // Clear all columns
        Object.values(columns).forEach(col => col.innerHTML = '');

        // Sort and render tasks
        const tasks = Object.values(this.state.tasks || {});
        tasks.sort((a, b) => a.task_id.localeCompare(b.task_id));

        for (const task of tasks) {
            const card = this.createTaskCard(task);
            const column = columns[task.status];
            if (column) {
                column.appendChild(card);
            }
        }
    }

    createTaskCard(task) {
        const card = document.createElement('div');
        card.className = 'task-card';

        // Get task title from tasks if available, otherwise show task_id
        const title = task.title || task.task_id;

        card.innerHTML = `
            <div class="task-id">${task.task_id}</div>
            <div class="task-title">${this.truncate(title, 40)}</div>
            <div class="task-meta">
                <span class="bounty" title="Bounty">💰 ${task.bounty_current}</span>
                ${task.fail_count > 0 ? `<span class="fails" title="Failed attempts">❌ ${task.fail_count}</span>` : ''}
            </div>
            ${task.assigned_worker ? `<div class="assigned-worker">👷 ${task.assigned_worker}</div>` : ''}
        `;
        return card;
    }

    renderWorkers() {
        const grid = document.getElementById('workers-grid');
        grid.innerHTML = '';

        const workers = Object.values(this.state.workers || {});
        workers.sort((a, b) => a.worker_id.localeCompare(b.worker_id));

        for (const worker of workers) {
            const card = this.createWorkerCard(worker);
            grid.appendChild(card);
        }
    }

    createWorkerCard(worker) {
        const card = document.createElement('div');
        card.className = 'worker-card' + (worker.assigned_task ? ' busy' : '');

        const balanceClass = worker.balance > 0 ? 'positive' : (worker.balance < 0 ? 'negative' : '');

        card.innerHTML = `
            <div class="worker-id">
                <span class="dot"></span>
                ${worker.worker_id}
            </div>
            <div class="worker-stats">
                <div class="stat">
                    <span>Rep</span>
                    <span class="stat-value">${worker.reputation.toFixed(2)}</span>
                </div>
                <div class="stat">
                    <span>Balance</span>
                    <span class="stat-value ${balanceClass}">${worker.balance.toFixed(1)}</span>
                </div>
                <div class="stat">
                    <span>Wins</span>
                    <span class="stat-value">${worker.wins}</span>
                </div>
                <div class="stat">
                    <span>Done</span>
                    <span class="stat-value">${worker.completions}</span>
                </div>
            </div>
            ${worker.assigned_task ? `<div class="current-task">Working on: ${worker.assigned_task}</div>` : ''}
        `;
        return card;
    }

    renderEvents() {
        const log = document.getElementById('events-log');

        // Render events in reverse order (newest first)
        const reversed = [...this.events].reverse();
        log.innerHTML = reversed.map(e => this.createEventItem(e)).join('');
    }

    createEventItem(event) {
        const type = event.type || 'unknown';
        const typeClass = type.toLowerCase().replace(/_/g, '_');
        const time = this.formatTime(event.ts);
        const info = this.getEventInfo(event);

        return `
            <div class="event-item">
                <span class="event-type ${typeClass}">${type.replace(/_/g, ' ')}</span>
                <span class="event-info">${info}</span>
                <span class="event-time">${time}</span>
            </div>
        `;
    }

    getEventInfo(event) {
        const p = event.payload || {};
        switch (event.type) {
            case 'run_created':
                return `Run ${event.run_id} started`;
            case 'worker_registered':
                return `Worker ${p.worker_id} joined`;
            case 'task_created':
                return `Task ${p.task_id} (bounty: ${p.bounty})`;
            case 'bid_submitted': {
                const bids = Array.isArray(p.bids) ? p.bids : [];
                if (!bids.length) return `${p.worker_id} submitted no bids`;
                const top = bids.slice(0, 2).map(b => {
                    const pSuccess = (b.self_assessed_p_success ?? '').toString();
                    return `${b.task_id}@${b.ask}${pSuccess ? ` (p=${pSuccess})` : ''}`;
                }).join(', ');
                const extra = bids.length > 2 ? ` +${bids.length - 2} more` : '';
                return `${p.worker_id} bids: ${top}${extra}`;
            }
            case 'task_assigned':
                return `${p.task_id} → ${p.worker_id}`;
            case 'market_cleared':
                const numAssignments = (p.assignments || []).length;
                return `${numAssignments} task${numAssignments !== 1 ? 's' : ''} assigned`;
            case 'round_advanced':
                return `{"round_id":${p.round_id}}`;
            case 'patch_submitted':
                return `${p.worker_id} submitted patch for ${p.task_id}`;
            case 'verification_passed':
                return `${p.task_id} passed verification`;
            case 'verification_failed':
                return `${p.task_id} failed verification`;
            case 'task_completed':
                return `${p.task_id} completed (${p.verify_status || 'PASS'})`;
            case 'payment_made':
                return `${p.worker_id} +${p.amount}`;
            case 'penalty_applied':
                return `${p.worker_id} -${p.amount}${p.reason ? ` (${p.reason})` : ''}`;
            case 'discussion_post':
                return `💬 ${p.sender}: "${this.truncate(p.message, 40)}"`;
            case 'plan_revision_requested':
                return `Plan revision requested (task ${p.task_id}, fails ${p.fail_count})`;
            default:
                return JSON.stringify(p).slice(0, 50);
        }
    }

    formatTime(ts) {
        if (!ts) return '--:--';
        const date = new Date(ts);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }

    truncate(str, len) {
        if (!str) return '';
        return str.length > len ? str.slice(0, len) + '...' : str;
    }
}

// Initialize dashboard on load
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
