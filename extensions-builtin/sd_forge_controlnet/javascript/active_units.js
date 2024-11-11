(function () {

    const delay = 2500;
    const ImageChangeType = Object.freeze({
        NO_CHANGE: 0,
        REMOVE: 1,
        ADD: 2,
        SRC_CHANGE: 3,
    });

    /** @param {MutationRecord[]} mutationsList @returns {ImageChangeType} */
    function imageChangeObserved(mutationsList) {
        for (const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                if (mutation.addedNodes.length > 0) {
                    for (const node of mutation.addedNodes) {
                        if (node.tagName === 'IMG')
                            return ImageChangeType.ADD;
                    }
                }
                if (mutation.removedNodes.length > 0) {
                    for (const node of mutation.removedNodes) {
                        if (node.tagName === 'IMG')
                            return ImageChangeType.REMOVE;
                    }
                }
            }
            else if (mutation.type === 'attributes') {
                if (mutation.target.tagName === 'IMG' && mutation.attributeName === 'src')
                    return ImageChangeType.SRC_CHANGE;
            }
        }
        return ImageChangeType.NO_CHANGE;
    }

    /** @param {HTMLElement} element @returns {number} */
    function childIndex(element) {
        return Array.from(element.parentNode.childNodes)
            .filter(child => child.nodeType === Node.ELEMENT_NODE)
            .indexOf(element);
    }

    class ControlNetUnit {

        constructor(unit) {
            /** @type {HTMLElement} */
            this.unit = unit;
            /** @type {number} */
            this.tabIndex = childIndex(unit);
            /** @type {boolean} */
            this.isT2I = !unit.querySelector('.cnet-mask-upload').id.includes('img2img');

            this.enabledCheckbox = unit.querySelector('.cnet-unit-enabled input');
            this.controlTypeRadios = unit.querySelectorAll('.controlnet_control_type_filter_group input[type="radio"]');
            this.inputImageContainer = unit.querySelector('.cnet-input-image-group .cnet-image');
            this.runPreprocessorButton = unit.querySelector('.cnet-run-preprocessor');

            this.#attachA1111SendInfoObserver();
            this.#attachControlTypeRadioListener();
            this.#attachEnabledButtonListener();
            this.#attachImageStateChangeObserver();
            this.#attachPresetDropdownObserver();

            /** @type {string} */
            this.suffix = "";
            /** @type {boolean} */
            this.enabled = false;
        }

        #attachA1111SendInfoObserver() {
            const pasteButton = document.getElementById(
                this.isT2I ? 'txt2img_tools' : 'img2img_tools'
            ).querySelector('#paste');

            const infoButtons = document.querySelectorAll(
                this.isT2I ? '#txt2img_tab' :
                    '#img2img_tab, #inpaint_tab'
            );

            for (const button of [pasteButton, ...infoButtons]) {
                button.addEventListener('click', () => {
                    setTimeout(() => { this.#updateActiveState(); }, delay);
                });
            }
        }

        #attachControlTypeRadioListener() {
            for (const radio of this.controlTypeRadios) {
                radio.addEventListener('change', () => {
                    this.#updateActiveControlType();
                });
            }
        }

        #attachEnabledButtonListener() {
            this.enabledCheckbox.addEventListener('change', () => {
                this.#updateActiveState();
            });
        }

        #attachImageStateChangeObserver() {
            const mo = new MutationObserver((mutationsList) => {
                const changeObserved = imageChangeObserved(mutationsList);

                if (changeObserved === ImageChangeType.ADD) {
                    this.runPreprocessorButton.removeAttribute("disabled");
                    this.runPreprocessorButton.title = 'Run Preprocessor';
                }
                else if (changeObserved === ImageChangeType.REMOVE) {
                    this.runPreprocessorButton.setAttribute("disabled", true);
                    this.runPreprocessorButton.title = "No Input Image";
                }
            });

            mo.observe(this.inputImageContainer, {
                childList: true,
                subtree: true,
            });
        }

        #attachPresetDropdownObserver() {
            const mo = new MutationObserver((mutationsList) => {
                for (const mutation of mutationsList) {
                    if (mutation.removedNodes.length > 0) {
                        setTimeout(() => {
                            this.#updateActiveState();
                            this.#updateActiveControlType();
                        }, delay);
                        break;
                    }
                }
            });

            const presetDropdown = this.unit.querySelector('.cnet-preset-dropdown');
            mo.observe(presetDropdown, {
                childList: true,
                subtree: true,
            });
        }

        #updateActiveState() {
            const tabHeader = this.#getTabHeader();
            this.enabled = this.enabledCheckbox.checked;
            if (this.enabled)
                tabHeader.classList.add('cnet-unit-active');
            else
                tabHeader.classList.remove('cnet-unit-active');
        }

        #updateActiveControlType() {
            const tabHeader = this.#getTabHeader();

            let controlTypeSuffix = tabHeader.querySelector('.control-type-suffix');
            if (controlTypeSuffix == null) {
                const span = document.createElement('span');
                span.classList.add('control-type-suffix');
                tabHeader.appendChild(span);
                controlTypeSuffix = span;
            }

            const controlType = this.#getActiveControlType();
            if (controlType === 'All')
                this.suffix = '';
            else
                this.suffix = `[${controlType}]`;

            controlTypeSuffix.textContent = this.suffix;
        }

        #getTabHeader() {
            return this.unit.parentElement.querySelectorAll('.tab-nav>button')[this.tabIndex - 1];
        }

        #getActiveControlType() {
            for (let radio of this.controlTypeRadios) {
                if (radio.checked) {
                    return radio.value;
                }
            }
            return undefined;
        }

        getStatus() {
            return [this.enabled, this.suffix];
        }
    }

    onUiLoaded(() => {
        document.querySelectorAll('#controlnet').forEach((ext) => {
            const units = [];
            const tabs = [...ext.querySelectorAll('.controlnet_tabs>.tabitem')];
            tabs.map((unit) => { units.push(new ControlNetUnit(unit)); });

            const nav = ext.querySelector(".tab-nav");

            function updateStatus() {
                for (const [i, btn] of Array.from(nav.querySelectorAll("button")).entries()) {
                    const [isOn, suffix] = units[i].getStatus();
                    if (isOn)
                        btn.classList.add('cnet-unit-active');
                    if (suffix) {
                        const span = document.createElement('span');
                        span.classList.add('control-type-suffix');
                        span.textContent = suffix;
                        btn.appendChild(span);
                    }
                }
            }

            nav.addEventListener("click", () => updateStatus());
        });
    });

})();
