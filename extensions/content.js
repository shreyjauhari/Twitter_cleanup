// ============================================================
// CONFIGURATION
// ============================================================

const MIN_CONFIDENCE = 60;


// ============================================================
// TRACK PROCESSED POSTS
// ============================================================

// WeakSet prevents the same post from being classified repeatedly
const processedPosts = new WeakSet();


// ============================================================
// GET BLOCKED CATEGORIES
// ============================================================

function getBlockedCategories() {

    return new Promise((resolve) => {

        chrome.storage.local.get(
            ["blockedCategories"],
            function (result) {

                resolve(
                    result.blockedCategories || []
                );

            }
        );

    });

}


// ============================================================
// SEND TEXT TO BACKGROUND SCRIPT
// ============================================================

// content.js does NOT call FastAPI directly.
// It sends the text to background.js.
// background.js then calls the AI backend.

async function classifyText(text) {

    return new Promise((resolve) => {

        chrome.runtime.sendMessage(

            {
                type: "CLASSIFY_TEXT",
                text: text
            },

            function (response) {

                // Check whether communication with background.js failed
                if (chrome.runtime.lastError) {

                    console.error(
                        "Extension error:",
                        chrome.runtime.lastError.message
                    );

                    resolve(null);

                    return;
                }


                // Check whether the backend returned an error
                if (
                    !response ||
                    !response.success
                ) {

                    console.error(
                        "Could not classify post:",
                        response?.error || "Unknown error"
                    );

                    resolve(null);

                    return;
                }


                // Return the prediction received from background.js
                resolve(response.data);

            }

        );

    });

}


// ============================================================
// PROCESS ONE POST
// ============================================================

async function processPost(post) {

    // Do not process the same post again
    if (processedPosts.has(post)) {

        return;

    }

    processedPosts.add(post);


    // --------------------------------------------------------
    // EXTRACT POST TEXT
    // --------------------------------------------------------

    const textElement = post.querySelector(
        '[data-testid="tweetText"]'
    );


    // Skip posts without text
    if (!textElement) {

        return;

    }


    const text = textElement.innerText.trim();


    // Skip empty or very short posts
    if (text.length < 5) {

        return;

    }


    console.log(
        "Classifying post:",
        text
    );


    // --------------------------------------------------------
    // GET AI PREDICTION
    // --------------------------------------------------------

    const prediction = await classifyText(text);


    // Stop if backend or extension failed
    if (!prediction) {

        return;

    }


    const category = prediction.category.toLowerCase();

    const confidence = prediction.confidence;


    console.log(
        "Prediction:",
        category,
        "| Confidence:",
        confidence + "%"
    );


    // --------------------------------------------------------
    // GET USER'S BLOCKED CATEGORIES
    // --------------------------------------------------------

    const blockedCategories =
        await getBlockedCategories();


    console.log(
        "Blocked categories:",
        blockedCategories
    );


    // --------------------------------------------------------
    // CHECK IF POST SHOULD BE HIDDEN
    // --------------------------------------------------------

    const shouldHide =

        blockedCategories.includes(category) &&

        confidence >= MIN_CONFIDENCE;


    if (shouldHide) {

        console.log(
            "Hiding post:",
            category
        );


        // Hide the complete X post
        post.style.display = "none";

    }

    else {

        console.log(
            "Post allowed:",
            category
        );

    }

}


// ============================================================
// SCAN PAGE FOR POSTS
// ============================================================

function scanPosts() {

    const posts = document.querySelectorAll(
        'article[data-testid="tweet"]'
    );


    posts.forEach(
        function (post) {

            processPost(post);

        }
    );

}


// ============================================================
// WATCH FOR NEW POSTS
// ============================================================

// X dynamically loads more posts while scrolling

const observer = new MutationObserver(
    function () {

        scanPosts();

    }
);


// Start observing the X page

observer.observe(
    document.body,
    {
        childList: true,
        subtree: true
    }
);


// ============================================================
// INITIAL SCAN
// ============================================================

console.log(
    "Twitter News Filter content script started"
);

scanPosts();