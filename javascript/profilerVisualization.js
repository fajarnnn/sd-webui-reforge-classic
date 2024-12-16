
function createRow(table, cellName, items) {
    let tr = document.createElement('tr');
    let res = [];

    items.forEach(function (x, i) {
        if (x === undefined) {
            res.push(null);
            return;
        }

        let td = document.createElement(cellName);
        td.textContent = x;
        tr.appendChild(td);
        res.push(td);

        let colspan = 1;
        for (let n = i + 1; n < items.length; n++) {
            if (items[n] !== undefined) {
                break;
            }

            colspan += 1;
        }

        if (colspan > 1) {
            td.colSpan = colspan;
        }
    });

    table.appendChild(tr);

    return res;
}

function showProfile(path, cutoff = 0.05) {
    requestGet(path, {}, function (data) {
        let table = document.createElement('table');
        table.className = 'popup-table';

        data.records['total'] = data.total;
        let keys = Object.keys(data.records).sort(function (a, b) {
            return data.records[b] - data.records[a];
        });
        let items = keys.map(function (x) {
            return { key: x, parts: x.split('/'), time: data.records[x] };
        });
        let maxLength = items.reduce(function (a, b) {
            return Math.max(a, b.parts.length);
        }, 0);

        let cols = createRow(table, 'th', ['record', 'seconds']);
        cols[0].colSpan = maxLength;

        function arraysEqual(a, b) {
            return !(a < b || b < a);
        }

        let addLevel = function (level, parent, hide) {
            let matching = items.filter(function (x) {
                return x.parts[level] && !x.parts[level + 1] && arraysEqual(x.parts.slice(0, level), parent);
            });
            let sorted = matching.sort(function (a, b) {
                return b.time - a.time;
            });
            let othersTime = 0;
            let othersList = [];
            let othersRows = [];
            let childrenRows = [];
            sorted.forEach(function (x) {
                let visible = x.time >= cutoff && !hide;

                let cells = [];
                for (let i = 0; i < maxLength; i++) {
                    cells.push(x.parts[i]);
                }
                cells.push(x.time.toFixed(3));
                let cols = createRow(table, 'td', cells);
                for (i = 0; i < level; i++) {
                    cols[i].className = 'muted';
                }

                let tr = cols[0].parentNode;
                if (!visible) {
                    tr.classList.add("hidden");
                }

                if (x.time >= cutoff) {
                    childrenRows.push(tr);
                } else {
                    othersTime += x.time;
                    othersList.push(x.parts[level]);
                    othersRows.push(tr);
                }

                let children = addLevel(level + 1, parent.concat([x.parts[level]]), true);
                if (children.length > 0) {
                    let cell = cols[level];
                    let onclick = function () {
                        cell.classList.remove("link");
                        cell.removeEventListener("click", onclick);
                        children.forEach(function (x) {
                            x.classList.remove("hidden");
                        });
                    };
                    cell.classList.add("link");
                    cell.addEventListener("click", onclick);
                }
            });

            if (othersTime > 0) {
                let cells = [];
                for (let i = 0; i < maxLength; i++) {
                    cells.push(parent[i]);
                }
                cells.push(othersTime.toFixed(3));
                cells[level] = 'others';
                let cols = createRow(table, 'td', cells);
                for (i = 0; i < level; i++) {
                    cols[i].className = 'muted';
                }

                let cell = cols[level];
                let tr = cell.parentNode;
                let onclick = function () {
                    tr.classList.add("hidden");
                    cell.classList.remove("link");
                    cell.removeEventListener("click", onclick);
                    othersRows.forEach(function (x) {
                        x.classList.remove("hidden");
                    });
                };

                cell.title = othersList.join(", ");
                cell.classList.add("link");
                cell.addEventListener("click", onclick);

                if (hide) {
                    tr.classList.add("hidden");
                }

                childrenRows.push(tr);
            }

            return childrenRows;
        };

        addLevel(0, []);

        popup(table);
    });
}
