// ============================================================
// GET HTML ELEMENTS
// ============================================================

const saveButton = document.getElementById("saveButton");

const status = document.getElementById("status");

const checkboxes = document.querySelectorAll(
    '.category input[type="checkbox"]'
);


// ============================================================
// LOAD SAVED FILTERS WHEN POPUP OPENS
// ============================================================

chrome.storage.local.get(
    ["blockedCategories"],
    function (result) {

        const blockedCategories =
            result.blockedCategories || [];

        // Check boxes that were previously saved
        checkboxes.forEach(function (checkbox) {

            if (
                blockedCategories.includes(
                    checkbox.value
                )
            ) {

                checkbox.checked = true;

            }

        });

    }
);


// ============================================================
// SAVE FILTERS
// ============================================================

saveButton.addEventListener(
    "click",
    function () {

        // Array that will store selected categories
        const blockedCategories = [];


        // Check every checkbox
        checkboxes.forEach(
            function (checkbox) {

                // If selected, add its category
                if (checkbox.checked) {

                    blockedCategories.push(
                        checkbox.value
                    );

                }

            }
        );


        // Save categories in Chrome extension storage
        chrome.storage.local.set(
            {
                blockedCategories:
                    blockedCategories
            },
            function () {

                // Show confirmation
                status.textContent =
                    "Filters saved successfully!";

                console.log(
                    "Blocked categories:",
                    blockedCategories
                );


                // Remove status message after 2 seconds
                setTimeout(
                    function () {

                        status.textContent = "";

                    },
                    2000
                );

            }
        );

    }
);