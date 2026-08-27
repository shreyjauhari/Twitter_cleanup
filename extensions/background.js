// ============================================================
// CONFIGURATION
// ============================================================

const API_URL = "http://127.0.0.1:8000/predict";


// ============================================================
// EXTENSION INSTALLED
// ============================================================

chrome.runtime.onInstalled.addListener(() => {

    console.log(
        "Twitter News Category Filter installed"
    );

});


// ============================================================
// RECEIVE TEXT FROM content.js
// AND SEND IT TO THE AI BACKEND
// ============================================================

chrome.runtime.onMessage.addListener(
    (message, sender, sendResponse) => {

        // Only handle classification requests
        if (message.type === "CLASSIFY_TEXT") {

            console.log(
                "Sending text to AI backend:",
                message.text
            );


            fetch(API_URL, {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: message.text
                })

            })

            .then(response => {

                if (!response.ok) {

                    throw new Error(
                        `API Error: ${response.status}`
                    );

                }

                return response.json();

            })

            .then(data => {

                console.log(
                    "AI prediction received:",
                    data
                );


                // Send prediction back to content.js
                sendResponse({

                    success: true,

                    data: data

                });

            })

            .catch(error => {

                console.error(
                    "Backend error:",
                    error
                );


                // Tell content.js that something failed
                sendResponse({

                    success: false,

                    error: error.message

                });

            });


            // IMPORTANT:
            // Keeps the message channel open because
            // fetch() is asynchronous.
            return true;

        }

    }
);