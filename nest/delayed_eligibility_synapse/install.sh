#!/usr/bin/env bash
# Install the compiled mymodule.so into the correct NEST module directory.
# Run this script from the ROOT of your module (the folder containing build/).

set -euo pipefail

MODULE_NAME="mymodule"
SO_FILE="${MODULE_NAME}.so"
BUILD_SO_PATH="build/src/${SO_FILE}"

# --- Step 1: Verify the compiled module exists ---
if [ ! -f "${BUILD_SO_PATH}" ]; then
    echo "❌ Error: ${BUILD_SO_PATH} not found."
    echo "Make sure you've built the module first, e.g.:"
    echo "    mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# --- Step 2: Locate NEST installation directory ---
if command -v nest-config &>/dev/null; then
    NEST_PREFIX=$(nest-config --prefix)
    NEST_LIBDIR="${NEST_PREFIX}/lib/nest"
else
    echo "⚠️  'nest-config' not found. Attempting to auto-detect NEST install directory..."

    for path in /usr/lib*/nest /usr/local/lib*/nest ~/.local/lib*/nest; do
        if [ -d "$path" ]; then
            NEST_LIBDIR="$path"
            break
        fi
    done

    if [ -z "${NEST_LIBDIR:-}" ]; then
        echo "❌ Could not find NEST library directory automatically."
        echo "Please edit this script and set NEST_LIBDIR manually."
        exit 1
    fi
fi

# --- Step 3: Verify target directory ---
if [ ! -d "${NEST_LIBDIR}" ]; then
    echo "❌ Error: NEST library directory not found at: ${NEST_LIBDIR}"
    exit 1
fi

# --- Step 4: Install the module ---
echo "📦 Installing ${BUILD_SO_PATH} → ${NEST_LIBDIR}/${SO_FILE}"
sudo cp "${BUILD_SO_PATH}" "${NEST_LIBDIR}/"
sudo chmod 644 "${NEST_LIBDIR}/${SO_FILE}"

# --- Step 5: Confirmation ---
echo "✅ Installed ${SO_FILE} successfully to ${NEST_LIBDIR}"
