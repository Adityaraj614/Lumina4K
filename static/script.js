const upscaleForm = document.getElementById("upscale-form");
const upscaleButton = document.getElementById("upscale-button");
const upscaleStatus = document.getElementById("upscale-status");
const singleImageInput = document.getElementById("single-image");
const scaleSelect = document.getElementById("scale");
const modeSuggestionText = document.getElementById("modeSuggestionText");
const modeAiRadio = document.getElementById("mode-ai");
const modeResizeRadio = document.getElementById("mode-resize");
const previewContainer = document.getElementById("preview-container") || document.getElementById("comparison-section");
const originalPreview = document.getElementById("original-preview");
const upscaledPreview = document.getElementById("upscaled-preview");
const downloadButton = document.getElementById("download-button");
const loadingText = document.getElementById("loadingText");
const comparisonSection = document.getElementById("comparison-section");
const demoPreview = document.querySelector(".demo-preview");
const demoPreviewImage = document.getElementById("preview-image");
const DEMO_IMAGE = "/static/images/demo_upscale.jpg";

const batchForm = document.getElementById("batch-form");
const batchButton = document.getElementById("batch-button");
const batchStatus = document.getElementById("batch-status");
const batchResultText = document.getElementById("batchResultText");
const downloadBatchBtn = document.getElementById("downloadBatchBtn");
const batchResultsContainer = document.getElementById("batch-results") || document.getElementById("batchResultsContainer");

function setStatus(element, message, type) {
    element.textContent = message;
    element.className = "status" + (type ? " " + type : "");
}

function getErrorMessage(data, fallbackMessage) {
    return data && data.error ? data.error : fallbackMessage;
}

let originalImageUrl = null;
let processedImageUrl = null;
let processedFilename = "upscaled_image";
let selectedImageDimensions = null;
window.batchDownloadUrl = null;
const scale8Option = scaleSelect ? scaleSelect.querySelector('option[value="8"]') : null;

function clearObjectUrl(url) {
    if (url) {
        URL.revokeObjectURL(url);
    }
}

function resetDemoPreviewImage() {
    if (!demoPreviewImage) {
        return;
    }

    demoPreviewImage.src = DEMO_IMAGE;
}

function getEstimatedOutputPixels() {
    if (!selectedImageDimensions || !scaleSelect) {
        return 0;
    }

    const scale = Number.parseInt(scaleSelect.value, 10);
    if (!scale) {
        return 0;
    }

    return selectedImageDimensions.width * scale * selectedImageDimensions.height * scale;
}

function getSelectionWarningMessage() {
    const estimatedPixels = getEstimatedOutputPixels();
    const warnings = [];

    if (scale8Option && scale8Option.disabled) {
        warnings.push("Large image detected. 8x scaling disabled for stability.");
    }

    if (estimatedPixels > 20000000) {
        warnings.push("This may take longer and generate a large file. Continue?");
    }

    return warnings.join(" ");
}

function updateProcessingWarning() {
    if (!upscaleStatus) {
        return;
    }

    if (upscaleStatus.classList.contains("success") || upscaleStatus.classList.contains("error")) {
        return;
    }

    setStatus(upscaleStatus, getSelectionWarningMessage(), "");
}

function updateScaleAvailability(file) {
    if (!scaleSelect || !scale8Option) {
        return;
    }

    if (!file) {
        scale8Option.disabled = false;
        selectedImageDimensions = null;
        return;
    }

    const probeUrl = URL.createObjectURL(file);
    const probeImage = new Image();

    probeImage.onload = () => {
        const isLargeImage = probeImage.width > 1500 || probeImage.height > 1500;
        selectedImageDimensions = {
            width: probeImage.width,
            height: probeImage.height
        };

        if (modeAiRadio && modeResizeRadio && modeSuggestionText && probeImage.width < 800) {
            modeAiRadio.checked = true;
            modeSuggestionText.innerText = "💡 Suggested: Enhance (AI) for better detail on small images.";
        } else if (modeAiRadio && modeResizeRadio && modeSuggestionText && probeImage.width > 1200) {
            modeResizeRadio.checked = true;
            modeSuggestionText.innerText = "💡 Suggested: Resize mode for better quality on large images.";
        } else if (modeSuggestionText) {
            modeSuggestionText.innerText = "💡 Both modes work. Choose based on preference.";
        }

        scale8Option.disabled = isLargeImage;

        if (isLargeImage && scaleSelect.value === "8") {
            scaleSelect.value = "4";
        }

        updateProcessingWarning();

        URL.revokeObjectURL(probeUrl);
    };

    probeImage.onerror = () => {
        scale8Option.disabled = false;
        selectedImageDimensions = null;
        if (modeSuggestionText) {
            modeSuggestionText.innerText = "";
        }
        URL.revokeObjectURL(probeUrl);
    };

    probeImage.src = probeUrl;
}

function resetPreview() {
    if (!previewContainer || !downloadButton || !originalPreview || !upscaledPreview) {
        return;
    }

    previewContainer.hidden = true;
    if (comparisonSection) {
        comparisonSection.style.display = "none";
    }
    downloadButton.hidden = true;
    originalPreview.removeAttribute("src");
    upscaledPreview.removeAttribute("src");
    if (demoPreview) {
        demoPreview.classList.remove("processing");
    }
    resetDemoPreviewImage();

    clearObjectUrl(originalImageUrl);
    clearObjectUrl(processedImageUrl);
    originalImageUrl = null;
    processedImageUrl = null;
}

function triggerDownload(url, filename) {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function renderBatchResults(results) {
    if (!batchResultsContainer) {
        return;
    }

    batchResultsContainer.innerHTML = "";

    if (!results || !results.length) {
        batchResultsContainer.hidden = true;
        return;
    }

    results.forEach((item, index) => {
        const card = document.createElement("article");
        card.className = "batch-result-card";

        const title = document.createElement("h3");
        title.textContent = item.filename || `Result ${index + 1}`;

        const compare = document.createElement("div");
        compare.className = "image-compare";

        const originalBox = document.createElement("div");
        originalBox.className = "image-box";
        const originalLabel = document.createElement("p");
        originalLabel.textContent = "Original";
        const originalImage = document.createElement("img");
        originalImage.src = item.original || item.original_url || "";
        originalImage.alt = `Original ${item.filename || index + 1}`;
        originalBox.appendChild(originalLabel);
        originalBox.appendChild(originalImage);

        const upscaledBox = document.createElement("div");
        upscaledBox.className = "image-box";
        const upscaledLabel = document.createElement("p");
        upscaledLabel.textContent = "Upscaled";
        const upscaledImage = document.createElement("img");
        upscaledImage.src = item.upscaled || item.upscaled_url || "";
        upscaledImage.alt = `Upscaled ${item.filename || index + 1}`;
        upscaledBox.appendChild(upscaledLabel);
        upscaledBox.appendChild(upscaledImage);

        compare.appendChild(originalBox);
        compare.appendChild(upscaledBox);

        const actionRow = document.createElement("div");
        actionRow.className = "batch-result-download";
        const downloadItemButton = document.createElement("button");
        downloadItemButton.type = "button";
        downloadItemButton.textContent = "Download Image";
        downloadItemButton.addEventListener("click", () => {
            triggerDownload(item.download_url, item.filename || `upscaled_${index + 1}.png`);
        });
        actionRow.appendChild(downloadItemButton);

        card.appendChild(title);
        card.appendChild(compare);
        card.appendChild(actionRow);
        batchResultsContainer.appendChild(card);
    });

    batchResultsContainer.hidden = false;
}

if (upscaleForm && upscaleButton && upscaleStatus && singleImageInput && scaleSelect && loadingText) {
    upscaleForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        resetPreview();

        if (getEstimatedOutputPixels() > 20000000) {
            const confirmed = window.confirm(
                "This may take longer and generate a large file. Continue?"
            );

            if (!confirmed) {
                updateProcessingWarning();
                return;
            }
        }

        const formData = new FormData(upscaleForm);
        upscaleButton.disabled = true;
        upscaleButton.textContent = "Processing...";
        loadingText.style.display = "block";
        setStatus(upscaleStatus, "Upscaling image...", "");
        if (demoPreview) {
            demoPreview.classList.add("processing");
        }

        try {
            const response = await fetch("/upscale", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(getErrorMessage(errorData, "Upscaling failed."));
            }

            const blob = await response.blob();
            const uploadedFile = singleImageInput.files[0];

            if (!uploadedFile) {
                throw new Error("Please select an image.");
            }

            originalImageUrl = URL.createObjectURL(uploadedFile);
            processedImageUrl = URL.createObjectURL(blob);
            processedFilename = "upscaled_" + uploadedFile.name;

            originalPreview.src = originalImageUrl;
            upscaledPreview.src = processedImageUrl;
            previewContainer.hidden = false;
            if (comparisonSection) {
                comparisonSection.style.display = "flex";
                comparisonSection.scrollIntoView({
                    behavior: "smooth"
                });
            }
            if (demoPreview) {
                demoPreview.classList.remove("processing");
            }
            resetDemoPreviewImage();
            downloadButton.hidden = false;

            setStatus(upscaleStatus, "Processing complete. Preview the result and download when ready.", "success");
        } catch (error) {
            setStatus(upscaleStatus, error.message, "error");
            if (demoPreview) {
                demoPreview.classList.remove("processing");
            }
        } finally {
            loadingText.style.display = "none";
            upscaleButton.disabled = false;
            upscaleButton.textContent = "Upscale Image";
        }
    });

    singleImageInput.addEventListener("change", () => {
        resetPreview();
        const file = singleImageInput.files[0];
        if (file && demoPreviewImage) {
            const reader = new FileReader();
            reader.onload = function(event) {
                demoPreviewImage.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
        updateScaleAvailability(singleImageInput.files[0]);
    });

    scaleSelect.addEventListener("change", () => {
        updateProcessingWarning();
    });
}

if (downloadButton) {
    downloadButton.addEventListener("click", () => {
        if (processedImageUrl) {
            triggerDownload(processedImageUrl, processedFilename);
        }
    });
}

if (downloadBatchBtn) {
    downloadBatchBtn.onclick = async () => {
        if (!window.batchDownloadUrl) {
            return;
        }

        try {
            const response = await fetch(window.batchDownloadUrl);
            if (!response.ok) {
                throw new Error("ZIP download failed.");
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            triggerDownload(url, "batch_results.zip");
            window.setTimeout(() => {
                window.URL.revokeObjectURL(url);
            }, 1000);
        } catch (error) {
            console.error(error);
            setStatus(batchStatus, error.message || "ZIP download failed.", "error");
        }
    };
}

if (batchForm && batchButton && batchStatus && batchResultText && downloadBatchBtn) {
    async function runBatchProcessing(event) {
        event.preventDefault();

        const batchInput = document.getElementById("batchInput") || document.getElementById("batch-images");
        const files = batchInput ? batchInput.files : [];
        if (!files.length) {
            setStatus(batchStatus, "Please select at least one image.", "error");
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i += 1) {
            formData.append("images", files[i]);
        }

        window.batchDownloadUrl = null;
        downloadBatchBtn.style.display = "none";
        batchResultText.innerText = "";
        renderBatchResults([]);

        batchButton.disabled = true;
        batchButton.textContent = "Processing...";
        setStatus(batchStatus, "Running batch processing...", "");

        try {
            const response = await fetch("/batch", {
                method: "POST",
                body: formData
            });

            const rawText = await response.text();
            console.log("RAW RESPONSE:", rawText);

            let data;
            try {
                data = JSON.parse(rawText);
            } catch (err) {
                console.error("Not JSON:", rawText);
                throw new Error("Server returned non-JSON response");
            }

            console.log("Parsed JSON:", data);

            if (!response.ok) {
                throw new Error(data.error || "Server error");
            }

            if (!data.success) {
                alert(data.error || "Batch failed");
                return;
            }

            if (!Array.isArray(data.results)) {
                throw new Error("Batch processing returned an unexpected response.");
            }

            batchResultText.innerText = `Processed: ${data.processed_count || 0} images, Skipped: ${data.skipped_count || 0} images`;
            if (data.skipped_files && data.skipped_files.length) {
                batchResultText.innerText += ` | Skipped files: ${data.skipped_files.join(", ")}`;
            }

            window.batchDownloadUrl = data.zip_url || data.zip_download_url || null;
            downloadBatchBtn.style.display = window.batchDownloadUrl ? "block" : "none";
            renderBatchResults(data.results);
            setStatus(batchStatus, "Batch processing complete. ZIP is ready to download.", "success");

            if (window.batchDownloadUrl) {
                const a = document.createElement("a");
                a.href = window.batchDownloadUrl;
                a.download = "batch_results.zip";
                document.body.appendChild(a);
                a.click();
                a.remove();
            }
        } catch (err) {
            console.error(err);
            renderBatchResults([]);
            setStatus(batchStatus, err.message || "Batch processing failed.", "error");
            alert(err.message);
        } finally {
            batchButton.disabled = false;
            batchButton.textContent = "Run Batch Processing";
        }
    }

    batchForm.addEventListener("submit", runBatchProcessing);
}

if (document.getElementById("filters-upload")) {
    const filtersUpload = document.getElementById("filters-upload");
    const originalFiltersImage = document.getElementById("originalImage");
    const filteredFiltersImage = document.getElementById("filteredImage");
    const filterButtons = document.querySelectorAll(".filter-btn");
    const downloadFilteredBtn = document.getElementById("downloadFilteredBtn");
    const filterLoading = document.getElementById("filterLoading");
    const resetFilterBtn = document.getElementById("resetFilterBtn");

    let currentFile = null;
    let filtersPreviewUrl = null;
    let filteredResultUrl = null;

    filtersUpload.addEventListener("change", () => {
        currentFile = filtersUpload.files[0];

        if (!currentFile) return;

        if (filtersPreviewUrl) URL.revokeObjectURL(filtersPreviewUrl);
        if (filteredResultUrl) URL.revokeObjectURL(filteredResultUrl);

        filtersPreviewUrl = URL.createObjectURL(currentFile);

        originalFiltersImage.src = filtersPreviewUrl;
        filteredFiltersImage.src = filtersPreviewUrl;
        downloadFilteredBtn.style.display = "none";

        filterButtons.forEach(btn => btn.classList.remove("active"));
    });

    filterButtons.forEach(button => {
        button.addEventListener("click", () => {
            const filter = button.dataset.filter;

            if (!currentFile) return;

            filterButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");

            const formData = new FormData();
            formData.append("image", currentFile);
            formData.append("filter", filter);

            filterLoading.style.display = "block";
            filterButtons.forEach(btn => btn.disabled = true);

            fetch("/apply-filter", {
                method: "POST",
                body: formData
            })
                .then(res => {
                    if (!res.ok) throw new Error("Filter failed");
                    return res.blob();
                })
                .then(blob => {
                    if (filteredResultUrl) URL.revokeObjectURL(filteredResultUrl);

                    filteredResultUrl = URL.createObjectURL(blob);
                    filteredFiltersImage.src = filteredResultUrl;
                    downloadFilteredBtn.style.display = "block";
                    filterLoading.style.display = "none";
                    filterButtons.forEach(btn => btn.disabled = false);
                })
                .catch(err => {
                    console.error(err);
                    filterLoading.style.display = "none";
                    filterButtons.forEach(btn => btn.disabled = false);
                });
        });
    });

    downloadFilteredBtn.addEventListener("click", () => {
        if (!filteredResultUrl) return;

        const a = document.createElement("a");
        a.href = filteredResultUrl;
        a.download = "filtered_image.png";
        a.click();
    });

    resetFilterBtn.addEventListener("click", () => {
        if (!filtersPreviewUrl) return;

        filteredFiltersImage.src = filtersPreviewUrl;
        filterButtons.forEach(btn => btn.classList.remove("active"));
        downloadFilteredBtn.style.display = "none";
    });
}

if (document.getElementById("contentUpload")) {
    const contentUpload = document.getElementById("contentUpload");
    const applyStyleBtn = document.getElementById("applyStyleBtn");
    const stylePreviewImage = document.getElementById("stylePreviewImage");
    const downloadBtn = document.getElementById("downloadBtn");
    const stylePreviewPlaceholder = document.getElementById("stylePreviewPlaceholder");
    const stylePreviewMessage = document.getElementById("stylePreviewMessage");
    const showOriginalStyleBtn = document.getElementById("showOriginalStyleBtn");
    const showStyledStyleBtn = document.getElementById("showStyledStyleBtn");
    const slider = document.getElementById("strength");
    const valueText = document.getElementById("strengthValue");
    const styleButtons = document.querySelectorAll(".style-card");

    let contentFile = null;
    let selectedStyle = "mosaic";
    let originalImage = "";
    let styledImage = "";
    let currentStyleView = "original";

    function setStylePreviewMessage(message) {
        if (!stylePreviewMessage) {
            return;
        }

        stylePreviewMessage.textContent = message;
    }

    function setStyleToggleState(view) {
        currentStyleView = view;
        if (showOriginalStyleBtn) {
            showOriginalStyleBtn.classList.toggle("active", view === "original");
        }
        if (showStyledStyleBtn) {
            showStyledStyleBtn.classList.toggle("active", view === "styled");
        }

        const targetImage = view === "styled" ? styledImage : originalImage;
        if (!targetImage || !stylePreviewImage) {
            return;
        }

        stylePreviewImage.classList.add("is-fading");
        window.setTimeout(() => {
            stylePreviewImage.src = targetImage;
            stylePreviewImage.hidden = false;
            if (stylePreviewPlaceholder) {
                stylePreviewPlaceholder.style.display = "none";
            }
            window.setTimeout(() => {
                stylePreviewImage.classList.remove("is-fading");
            }, 20);
        }, 120);
    }

    function resetStyledPreview() {
        styledImage = "";
        if (showStyledStyleBtn) {
            showStyledStyleBtn.disabled = true;
        }
        if (downloadBtn) {
            downloadBtn.style.display = "none";
            downloadBtn.removeAttribute("href");
        }
        setStylePreviewMessage("Apply a style to preview result.");
        if (currentStyleView === "styled") {
            setStyleToggleState("original");
        }
    }

    contentUpload.addEventListener("change", () => {
        contentFile = contentUpload.files[0];
        if (contentFile) {
            clearObjectUrl(originalImage);
            originalImage = URL.createObjectURL(contentFile);
            resetStyledPreview();
            if (stylePreviewImage) {
                stylePreviewImage.src = originalImage;
                stylePreviewImage.hidden = false;
                stylePreviewImage.classList.remove("is-fading");
            }
            if (stylePreviewPlaceholder) {
                stylePreviewPlaceholder.style.display = "none";
            }
            setStyleToggleState("original");
        }
    });

    styleButtons.forEach(button => {
        button.addEventListener("click", () => {
            selectedStyle = button.dataset.style || "mosaic";
            styleButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
        });
    });

    if (slider && valueText) {
        slider.oninput = () => {
            valueText.innerText = Math.round(Number.parseFloat(slider.value) * 100) + "%";
        };
    }

    applyStyleBtn.addEventListener("click", () => {
        if (!contentFile) {
            alert("Upload a content image");
            return;
        }

        const formData = new FormData();
        formData.append("content", contentFile);
        formData.append("style", selectedStyle);
        if (slider) {
            formData.append("strength", slider.value);
        }

        applyStyleBtn.disabled = true;
        applyStyleBtn.textContent = "Applying...";
        if (stylePreviewImage) {
            stylePreviewImage.classList.add("is-fading");
        }
        setStylePreviewMessage("Processing style transfer...");

        fetch("/style-transfer", {
            method: "POST",
            body: formData
        })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(err => {
                        throw new Error(err.error || "Style transfer failed");
                    });
                }
                return res.blob();
            })
            .then(blob => {
                if (styledImage && styledImage.startsWith("blob:")) {
                    URL.revokeObjectURL(styledImage);
                }
                styledImage = URL.createObjectURL(blob);
                if (stylePreviewPlaceholder) {
                    stylePreviewPlaceholder.style.display = "none";
                }
                if (showStyledStyleBtn) {
                    showStyledStyleBtn.disabled = false;
                }
                setStylePreviewMessage("Styled preview ready.");
                setStyleToggleState("styled");
                if (downloadBtn) {
                    downloadBtn.style.display = "inline-block";
                    downloadBtn.href = styledImage;
                }
                applyStyleBtn.disabled = false;
                applyStyleBtn.textContent = "Apply Style";
            })
            .catch(err => {
                console.error(err);
                applyStyleBtn.disabled = false;
                applyStyleBtn.textContent = "Apply Style";
                if (stylePreviewImage) {
                    stylePreviewImage.classList.remove("is-fading");
                }
                setStylePreviewMessage("Apply a style to preview result.");
                alert(err.message);
            });
    });

    if (showOriginalStyleBtn) {
        showOriginalStyleBtn.addEventListener("click", () => {
            if (!originalImage) {
                setStylePreviewMessage("Upload an image to preview the original.");
                return;
            }
            setStyleToggleState("original");
        });
    }

    if (showStyledStyleBtn) {
        showStyledStyleBtn.addEventListener("click", () => {
            if (!styledImage) {
                setStylePreviewMessage("Apply a style to preview result.");
                return;
            }
            setStyleToggleState("styled");
        });
    }

    window.addEventListener("beforeunload", () => {
        clearObjectUrl(originalImage);
        if (!styledImage || styledImage.startsWith("/")) {
            return;
        }
        clearObjectUrl(styledImage);
    });
}
