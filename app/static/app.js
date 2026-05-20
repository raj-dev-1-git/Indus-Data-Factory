const uploadBtn = document.getElementById(
    "uploadBtn"
);

const audioFile = document.getElementById(
    "audioFile"
);

const statusDiv = document.getElementById(
    "status"
);

uploadBtn.addEventListener(

    "click",

    async () => {

        const file = audioFile.files[0];

        if (!file) {

            statusDiv.innerHTML =
                "<p>Select a WAV file.</p>";

            return;
        }

        const formData = new FormData();

        formData.append(
            "file",
            file
        );

        statusDiv.innerHTML = `

            <div class="loading">
                Running intelligence pipeline...
            </div>
        `;

        // =====================
        // API
        // =====================

        const response = await fetch(

            "/upload",

            {
                method: "POST",

                body: formData
            }
        );

        const data = await response.json();

        // =====================
        // ERROR
        // =====================

        if (!response.ok) {

            statusDiv.innerHTML = `

                <div class="card">

                    <h2>Error</h2>

                    <p>
                        ${data.error}
                    </p>

                </div>
            `;

            return;
        }

        // =====================
        // DATA
        // =====================

        const perception =
            data.perception;

        const reasoning =
            data.reasoning || {};

        // =====================
        // EVENTS
        // =====================

        let eventsHtml = "";

        if (
            perception.events &&
            perception.events.length > 0
        ) {

            for (
                const event
                of perception.events
            ) {

                eventsHtml += `

                    <div class="event">

                        <div class="event-title">
                            ${event.content}
                        </div>

                        <div class="event-meta">
                            ${event.time}
                        </div>

                        <div class="event-meta">
                            confidence:
                            ${event.confidence}
                        </div>

                    </div>
                `;
            }
        }

        else {

            eventsHtml =
                "<p>No events detected.</p>";
        }

        // =====================
        // SAFE VALUES
        // =====================

        const summary =
            reasoning.summary ||
            "No summary available.";

        const environment =

            typeof reasoning.environment ===
            "string"

                ? reasoning.environment

                : JSON.stringify(

                    reasoning.environment,

                    null,

                    2
                );

        const inference =
            reasoning.inference ||
            "No inference available.";

        const risk =
            reasoning.risk_level ||
            "Unknown";

        // =====================
        // UI
        // =====================

        statusDiv.innerHTML = `

            <div class="card">

                <h2>Transcript</h2>

                <p>
                    ${perception.full_transcript}
                </p>

            </div>

            <div class="card">

                <h2>Detected Events</h2>

                ${eventsHtml}

            </div>

            <div class="card">

                <h2>Environment</h2>

                <p>
                    ${environment}
                </p>

            </div>

            <div class="card">

                <h2>Summary</h2>

                <p>
                    ${summary}
                </p>

            </div>

            <div class="card">

                <h2>Inference</h2>

                <p>
                    ${inference}
                </p>

            </div>

            <div class="card">

                <h2>Risk Level</h2>

                <p>
                    ${risk}
                </p>

            </div>
        `;
    }
);

// =====================
// HISTORY
// =====================

const historyBtn = document.getElementById("historyBtn");
const historyContainer = document.getElementById("historyContainer");

historyBtn.addEventListener("click", async () => {
    historyContainer.innerHTML = '<div class="loading">Loading history...</div>';
    try {
        const response = await fetch("/history");
        const data = await response.json();
        
        if (!data.history || data.history.length === 0) {
            historyContainer.innerHTML = '<p>No history found.</p>';
            return;
        }
        
        let html = "";
        for (const record of data.history) {
            html += `
                <div class="card history-card">
                    <h3>${record.filename}</h3>
                    <p><strong>Risk:</strong> ${record.risk_level}</p>
                    <p class="history-summary">${record.summary}</p>
                </div>
            `;
        }
        historyContainer.innerHTML = html;
    } catch (e) {
        historyContainer.innerHTML = `<p>Error loading history: ${e}</p>`;
    }
});