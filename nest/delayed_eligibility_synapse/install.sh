#!/usr/bin/env bash
# Install the compiled mymodule.so into the correct NEST module directory.
# Run this script from the ROOT of your module (the folder containing build/).

set -euo pipefail

MODULE_NAME="mymodule"
SO_FILE="${MODULE_NAME}.so"
BUILD_SO_PATH="build/src/${SO_FILE}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Step 1: Verify the compiled module exists ---
if [ ! -f "${BUILD_SO_PATH}" ]; then
    echo "❌ Error: ${BUILD_SO_PATH} not found."
    echo "Make sure you've built the module first, e.g.:"
    echo "    mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# --- Step 2: Locate NEST installation directory ---
NEST_CONFIG_BIN="${NEST_CONFIG_BIN:-}"
if [ -z "${NEST_CONFIG_BIN}" ]; then
    if command -v nest-config >/dev/null 2>&1; then
        NEST_CONFIG_BIN="$(command -v nest-config)"
    elif [ -x "${SCRIPT_DIR}/../nest-install/bin/nest-config" ]; then
        NEST_CONFIG_BIN="${SCRIPT_DIR}/../nest-install/bin/nest-config"
    fi
fi

if [ -n "${NEST_CONFIG_BIN}" ]; then
    NEST_PREFIX="$("${NEST_CONFIG_BIN}" --prefix)"
    NEST_LIBDIR="${NEST_LIBDIR:-${NEST_PREFIX}/lib/nest}"
else
    echo "⚠️  'nest-config' not found in PATH. Attempting to auto-detect NEST install directory..."
    for path in \
        "${SCRIPT_DIR}/../nest-install/lib/nest" \
        /usr/lib*/nest \
        /usr/local/lib*/nest \
        "${HOME}"/.local/lib*/nest; do
        if [ -d "$path" ]; then
            NEST_LIBDIR="$path"
            break
        fi
    done
fi

if [ -z "${NEST_LIBDIR:-}" ]; then
    echo "❌ Could not determine NEST module directory."
    echo "Set it explicitly, e.g.:"
    echo "    NEST_LIBDIR=/path/to/nest/lib/nest ./install.sh"
    exit 1
fi

# --- Step 3: Ensure target directory exists ---
if [ ! -d "${NEST_LIBDIR}" ]; then
    echo "ℹ️  Creating NEST module directory: ${NEST_LIBDIR}"
    mkdir -p "${NEST_LIBDIR}"
fi

# --- Step 4: Install the module ---
echo "📦 Installing ${BUILD_SO_PATH} → ${NEST_LIBDIR}/${SO_FILE}"
if [ -w "${NEST_LIBDIR}" ]; then
    cp "${BUILD_SO_PATH}" "${NEST_LIBDIR}/"
    chmod 644 "${NEST_LIBDIR}/${SO_FILE}"
else
    sudo cp "${BUILD_SO_PATH}" "${NEST_LIBDIR}/"
    sudo chmod 644 "${NEST_LIBDIR}/${SO_FILE}"
fi

# --- Step 5: Confirmation ---
echo "✅ Installed ${SO_FILE} successfully to ${NEST_LIBDIR}"
