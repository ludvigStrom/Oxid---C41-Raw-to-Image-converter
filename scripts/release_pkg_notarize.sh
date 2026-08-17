#!/usr/bin/env bash
set -euo pipefail

# Build, package, notarize, and staple a macOS installer for Oxid.
#
# Usage:
#   bash scripts/release_pkg_notarize.sh            # version from Cargo.toml
#   bash scripts/release_pkg_notarize.sh 0.1.0
#
# Signed installer only (no notarytool / stapler):
#   SKIP_NOTARIZATION=1 bash scripts/release_pkg_notarize.sh
#
# Universal binary (arm64 + x86_64):
#   UNIVERSAL=1 bash scripts/release_pkg_notarize.sh
#
# Apple Team ID, bundle id, and notary profile live in .env (gitignored).
#   cp .env.example .env
# Store notary credentials once:
#   xcrun notarytool store-credentials "$NOTARY_PROFILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load KEY=value from .env without executing it. Existing env vars win.
load_dotenv () {
  local file="$1"
  [[ -f "${file}" ]] || return 0
  local line key val
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    if [[ "${line}" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      if [[ "${val}" =~ ^\"(.*)\"$ ]]; then
        val="${BASH_REMATCH[1]}"
      elif [[ "${val}" =~ ^\'(.*)\'$ ]]; then
        val="${BASH_REMATCH[1]}"
      fi
      if [[ -z "${!key+x}" ]]; then
        printf -v "${key}" '%s' "${val}"
        export "${key}"
      fi
    fi
  done < "${file}"
}

ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"
load_dotenv "${ENV_FILE}"

cargo_toml_version () {
  sed -n 's/^version = "\([^"]*\)"/\1/p' "${PROJECT_DIR}/Cargo.toml" | head -n 1
}

VERSION="${1:-$(cargo_toml_version)}"
APP_NAME="${APP_NAME:-Oxid}"
# Short slug for .pkg filenames (no spaces)
PKG_FILE_BASENAME="${PKG_FILE_BASENAME:-Oxid}"
PKG_ID_PREFIX="${PKG_ID_PREFIX:-}"
INSTALLER_TITLE="${INSTALLER_TITLE:-${APP_NAME}}"
BUNDLE_ID="${BUNDLE_ID:-${PKG_ID_PREFIX}}"
# Binary name inside Contents/MacOS (no spaces)
APP_EXECUTABLE="${APP_EXECUTABLE:-c41-gui}"
CARGO_BIN="${CARGO_BIN:-c41-gui}"
CARGO_FEATURES="${CARGO_FEATURES:-gui,gpu}"
UNIVERSAL="${UNIVERSAL:-0}"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"

TEAM_ID="${TEAM_ID:-}"
APP_SIGN_IDENTITY="${APP_SIGN_IDENTITY:-Developer ID Application}"
INSTALLER_SIGN_IDENTITY="${INSTALLER_SIGN_IDENTITY:-Developer ID Installer}"
APP_SIGN_IDENTITY_HASH="${APP_SIGN_IDENTITY_HASH:-}"
INSTALLER_SIGN_IDENTITY_HASH="${INSTALLER_SIGN_IDENTITY_HASH:-}"
NOTARY_PROFILE="${NOTARY_PROFILE:-}"

STAPLE_RETRIES="${STAPLE_RETRIES:-24}"
STAPLE_RETRY_DELAY_SEC="${STAPLE_RETRY_DELAY_SEC:-}"
STAPLE_RETRY_RAMP_MIN_SEC="${STAPLE_RETRY_RAMP_MIN_SEC:-60}"
STAPLE_RETRY_RAMP_MAX_SEC="${STAPLE_RETRY_RAMP_MAX_SEC:-300}"
POST_NOTARIZE_SLEEP_SEC="${POST_NOTARIZE_SLEEP_SEC:-60}"
STAPLE_TMP_COPY="${STAPLE_TMP_COPY:-}"
STAPLE_TMP_DIR="${STAPLE_TMP_DIR:-${TMPDIR:-/tmp}}"
STAPLE_ABORT_ON_C="${STAPLE_ABORT_ON_C:-1}"
SKIP_NOTARIZATION="${SKIP_NOTARIZATION:-0}"
STAPLE_VERBOSE="${STAPLE_VERBOSE:-0}"
STAPLE_VERBOSE_AFTER_FAIL="${STAPLE_VERBOSE_AFTER_FAIL:-1}"

LOGO_PNG="${LOGO_PNG:-${PROJECT_DIR}/src/img/logo.png}"
LICENSE_SRC="${LICENSE_SRC:-${PROJECT_DIR}/LICENSE}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This script only runs on macOS." >&2
  exit 1
fi

if [[ -z "${VERSION}" ]]; then
  echo "ERROR: Could not read version from Cargo.toml (or pass it as argv[1])." >&2
  exit 1
fi

if [[ -z "${PKG_ID_PREFIX}" ]]; then
  echo "ERROR: PKG_ID_PREFIX is not set (bundle / installer id)." >&2
  echo "Copy .env.example to .env and fill in your Apple IDs, or export PKG_ID_PREFIX." >&2
  exit 1
fi

if [[ "${SKIP_NOTARIZATION}" != "1" ]]; then
  if [[ -z "${TEAM_ID}" || -z "${NOTARY_PROFILE}" ]]; then
    echo "ERROR: TEAM_ID and NOTARY_PROFILE are required for notarization." >&2
    echo "Copy .env.example to .env and fill them in, or set SKIP_NOTARIZATION=1." >&2
    exit 1
  fi
fi

volume_fstype () {
  local path="$1"
  local mount_point
  mount_point="$(df -P "${path}" | awk 'NR==2 { print $NF }')"
  mount | awk -v mp="${mount_point}" '
    index($0, " on " mp " ") {
      if (match($0, /\([^)]+\)/)) {
        inner = substr($0, RSTART + 1, RLENGTH - 2)
        split(inner, parts, ",")
        print parts[1]
      }
    }'
}

fstype_ok_for_codesign () {
  case "$1" in
    apfs|hfs|hfs+) return 0 ;;
    *) return 1 ;;
  esac
}

PROJECT_FSTYPE="$(volume_fstype "${PROJECT_DIR}")"
# Signing / pkgbuild / stapler need a local Apple filesystem. This repo may live on exFAT.
PACKAGING_ROOT="${PACKAGING_ROOT:-}"
if [[ -z "${PACKAGING_ROOT}" ]]; then
  if fstype_ok_for_codesign "${PROJECT_FSTYPE}"; then
    PACKAGING_ROOT="${PROJECT_DIR}/build"
  else
    PACKAGING_ROOT="${HOME}/Library/Caches/oxid-release"
    echo "WARN: Project volume is '${PROJECT_FSTYPE}' (not APFS/HFS)."
    echo "      Packaging/signing on local disk: ${PACKAGING_ROOT}"
  fi
fi

DIST_DIR="${PACKAGING_ROOT}/dist"
COMPONENT_PKG_DIR="${DIST_DIR}/component_pkgs"
PAYLOAD_APP_ROOT="${DIST_DIR}/payload_app"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"
APP_ENTITLEMENTS="${DIST_DIR}/Oxid.release.entitlements"
APP_COMPONENT_PLIST="${DIST_DIR}/AppComponent.plist"
APP_COMPONENT_PKG="${COMPONENT_PKG_DIR}/${PKG_FILE_BASENAME}-App-${VERSION}.pkg"
DIST_XML="${DIST_DIR}/Distribution.xml"
FINAL_PKG_PATH="${DIST_DIR}/${PKG_FILE_BASENAME}-${VERSION}.pkg"
LOCAL_DIST_DIR="${PROJECT_DIR}/build/dist"
LOCAL_FINAL_PKG_PATH="${LOCAL_DIST_DIR}/${PKG_FILE_BASENAME}-${VERSION}.pkg"

pkg_sha256 () {
  shasum -a 256 "$1" | awk '{ print $1 }'
}

resolve_codesign_identity_hash () {
  local wanted="$1"
  local usage="$2"
  local policy="${3:-codesigning}"
  local matches=()

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    matches+=("${line}")
  done < <(security find-identity -v -p "${policy}" \
    | awk -v wanted="${wanted}" '
        /^[[:space:]]*[0-9]+\)/ {
          hash=$2;
          line=$0;
          q1=index(line, "\"");
          q2=0;
          if (q1 > 0) {
            rest=substr(line, q1 + 1);
            q2=index(rest, "\"");
          }
          if (q1 > 0 && q2 > 0) {
            name=substr(line, q1 + 1, q2 - 1);
            if (index(name, wanted) > 0)
              print hash;
          }
        }')

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "ERROR: Could not resolve ${usage} identity matching: ${wanted}" >&2
    echo "Available identities for policy '${policy}':" >&2
    security find-identity -v -p "${policy}" >&2 || true
    if [[ "${usage}" == "installer signing" ]]; then
      echo >&2
      echo "You need a 'Developer ID Installer' certificate in your login keychain." >&2
      echo "Create/download it in Apple Developer Certificates, then import into Keychain Access." >&2
    fi
    exit 1
  fi

  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "WARN: Multiple ${usage} identities matched '${wanted}'. Using first hash: ${matches[0]}" >&2
  fi

  echo "${matches[0]}"
}

format_clock_time_from_now_secs () {
  local add_secs="$1"
  date -v+${add_secs}S "+%H:%M"
}

staple_wait_secs_after_failed_attempt () {
  local failed_attempt="$1"
  if [[ -n "${STAPLE_RETRY_DELAY_SEC}" ]]; then
    echo "${STAPLE_RETRY_DELAY_SEC}"
    return
  fi
  local min_s="${STAPLE_RETRY_RAMP_MIN_SEC}"
  local max_s="${STAPLE_RETRY_RAMP_MAX_SEC}"
  local exp=$(( failed_attempt - 1 ))
  local mult=1
  local i
  for (( i = 0; i < exp; i++ )); do
    mult=$(( mult * 2 ))
  done
  local d=$(( min_s * mult ))
  if (( d > max_s )); then
    d="${max_s}"
  fi
  echo "${d}"
}

sleep_with_optional_abort () {
  local total_secs="$1"
  local remaining="${total_secs}"

  if [[ ! -t 0 ]] || [[ "${STAPLE_ABORT_ON_C}" != "1" ]]; then
    sleep "${remaining}"
    return 0
  fi

  while (( remaining > 0 )); do
    if read -r -t 1 -n 1 key 2>/dev/null; then
      if [[ "${key}" == "c" || "${key}" == "C" ]]; then
        return 1
      fi
    fi
    remaining=$(( remaining - 1 ))
  done
  return 0
}

write_staple_status_file () {
  local status_line="$1"
  local path="${DIST_DIR}/last_release_staple_status.txt"
  {
    echo "$(date "+%Y-%m-%d %H:%M:%S %z")  ${status_line}"
    echo "  installer: ${FINAL_PKG_PATH}"
    if [[ -n "${SUBMISSION_ID:-}" ]]; then
      echo "  notary submission id: ${SUBMISSION_ID}"
    fi
  } >> "${path}"
}

run_manual_staple_sequence () {
  local pkg_path="$1"
  shift
  local -a staple_extra=()
  if [[ $# -gt 0 ]]; then
    staple_extra=("$@")
  fi

  if [[ ${#staple_extra[@]} -gt 0 ]]; then
    echo "    — stapler staple ${staple_extra[*]} \"${pkg_path}\""
    xcrun stapler staple "${staple_extra[@]}" "${pkg_path}"
  else
    echo "    — stapler staple \"${pkg_path}\""
    xcrun stapler staple "${pkg_path}"
  fi
  local staple_ec=$?
  if [[ "${staple_ec}" -ne 0 ]]; then
    return "${staple_ec}"
  fi
  echo "    — stapler validate \"${pkg_path}\""
  xcrun stapler validate "${pkg_path}"
}

staple_pkg_with_optional_tmp_copy () {
  local dest_pkg="$1"
  shift
  local -a staple_extra=()
  if [[ $# -gt 0 ]]; then
    staple_extra=("$@")
  fi

  if [[ "${STAPLE_TMP_COPY}" != "1" ]]; then
    run_manual_staple_sequence "${dest_pkg}" ${staple_extra[@]+"${staple_extra[@]}"}
    return
  fi

  local tmp_pkg
  tmp_pkg="$(mktemp "${STAPLE_TMP_DIR}/c41raw-staple.XXXXXX.pkg")" || return 1
  echo "    (stapling via temp file: ${tmp_pkg})"
  if ! ditto "${dest_pkg}" "${tmp_pkg}"; then
    rm -f "${tmp_pkg}"
    return 1
  fi
  if run_manual_staple_sequence "${tmp_pkg}" ${staple_extra[@]+"${staple_extra[@]}"}; then
    if ditto "${tmp_pkg}" "${dest_pkg}"; then
      rm -f "${tmp_pkg}"
      return 0
    fi
    rm -f "${tmp_pkg}"
    return 1
  fi
  rm -f "${tmp_pkg}"
  return 1
}

ensure_rust_target () {
  local triple="$1"
  if ! rustup target list --installed | grep -qx "${triple}"; then
    echo "==> Installing Rust target ${triple}" >&2
    rustup target add "${triple}"
  fi
}

built_bin_path () {
  local triple="${1:-}"
  if [[ -n "${triple}" ]]; then
    echo "${PROJECT_DIR}/target/${triple}/release/${CARGO_BIN}"
  else
    echo "${PROJECT_DIR}/target/release/${CARGO_BIN}"
  fi
}

build_gui_binary () {
  export MACOSX_DEPLOYMENT_TARGET
  local -a cargo_args=(build --release --bin "${CARGO_BIN}" --features "${CARGO_FEATURES}")

  if [[ "${UNIVERSAL}" == "1" ]]; then
    echo "==> Building universal ${CARGO_BIN} (${CARGO_FEATURES})" >&2
    ensure_rust_target aarch64-apple-darwin
    ensure_rust_target x86_64-apple-darwin
    cargo "${cargo_args[@]}" --target aarch64-apple-darwin
    cargo "${cargo_args[@]}" --target x86_64-apple-darwin
    local out="${DIST_DIR}/${CARGO_BIN}"
    lipo -create \
      -output "${out}" \
      "$(built_bin_path aarch64-apple-darwin)" \
      "$(built_bin_path x86_64-apple-darwin)"
    echo "${out}"
    return
  fi

  echo "==> Building ${CARGO_BIN} (${CARGO_FEATURES}, host)" >&2
  cargo "${cargo_args[@]}"
  local src
  src="$(built_bin_path)"
  local out="${DIST_DIR}/${CARGO_BIN}"
  ditto "${src}" "${out}"
  echo "${out}"
}

write_app_entitlements () {
  cat > "${APP_ENTITLEMENTS}" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.get-task-allow</key>
  <false/>
</dict>
</plist>
EOF
}

make_icns_from_png () {
  local png="$1"
  local icns="$2"
  local iconset
  iconset="$(mktemp -d "${DIST_DIR}/AppIcon.XXXXXX.iconset")"

  sips -z 16 16     "${png}" --out "${iconset}/icon_16x16.png" >/dev/null
  sips -z 32 32     "${png}" --out "${iconset}/icon_16x16@2x.png" >/dev/null
  sips -z 32 32     "${png}" --out "${iconset}/icon_32x32.png" >/dev/null
  sips -z 64 64     "${png}" --out "${iconset}/icon_32x32@2x.png" >/dev/null
  sips -z 128 128   "${png}" --out "${iconset}/icon_128x128.png" >/dev/null
  sips -z 256 256   "${png}" --out "${iconset}/icon_128x128@2x.png" >/dev/null
  sips -z 256 256   "${png}" --out "${iconset}/icon_256x256.png" >/dev/null
  sips -z 512 512   "${png}" --out "${iconset}/icon_256x256@2x.png" >/dev/null
  sips -z 512 512   "${png}" --out "${iconset}/icon_512x512.png" >/dev/null
  sips -z 1024 1024 "${png}" --out "${iconset}/icon_512x512@2x.png" >/dev/null
  iconutil -c icns "${iconset}" -o "${icns}"
  rm -rf "${iconset}"
}

write_info_plist () {
  local plist="$1"
  local has_icon="$2"
  local icon_keys=""
  if [[ "${has_icon}" == "1" ]]; then
    icon_keys="  <key>CFBundleIconFile</key>
  <string>AppIcon</string>"
  fi

  cat > "${plist}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleExecutable</key>
  <string>${APP_EXECUTABLE}</string>
${icon_keys}
  <key>CFBundleIdentifier</key>
  <string>${BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${VERSION}</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.photography</string>
  <key>LSMinimumSystemVersion</key>
  <string>${MACOSX_DEPLOYMENT_TARGET}</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
EOF
}

assemble_app_bundle () {
  local binary="$1"
  rm -rf "${APP_BUNDLE}"
  mkdir -p "${APP_BUNDLE}/Contents/MacOS" "${APP_BUNDLE}/Contents/Resources"

  ditto "${binary}" "${APP_BUNDLE}/Contents/MacOS/${APP_EXECUTABLE}"
  chmod 755 "${APP_BUNDLE}/Contents/MacOS/${APP_EXECUTABLE}"

  local has_icon=0
  if [[ -f "${LOGO_PNG}" ]]; then
    echo "==> Building app icon from ${LOGO_PNG}"
    make_icns_from_png "${LOGO_PNG}" "${APP_BUNDLE}/Contents/Resources/AppIcon.icns"
    has_icon=1
  else
    echo "WARN: Logo not found at ${LOGO_PNG}; app will use the default icon."
  fi

  write_info_plist "${APP_BUNDLE}/Contents/Info.plist" "${has_icon}"
}

assert_no_get_task_allow () {
  local binary="$1"
  local entitlements_tmp
  entitlements_tmp="$(mktemp "${DIST_DIR}/entitlements.XXXXXX.plist")"

  if [[ ! -f "${binary}" ]]; then
    echo "ERROR: Expected binary not found for entitlement check: ${binary}" >&2
    exit 1
  fi

  if ! codesign -d --entitlements :- "${binary}" > "${entitlements_tmp}" 2>/dev/null; then
    rm -f "${entitlements_tmp}"
    echo "ERROR: Could not read entitlements from: ${binary}" >&2
    exit 1
  fi

  local gta_value
  gta_value="$(/usr/libexec/PlistBuddy -c "Print :com.apple.security.get-task-allow" "${entitlements_tmp}" 2>/dev/null || true)"

  if [[ "${gta_value}" == "true" ]]; then
    rm -f "${entitlements_tmp}"
    echo "ERROR: get-task-allow entitlement detected in release binary:" >&2
    echo "  ${binary}" >&2
    echo "Fix signing settings before notarization (Release must not include get-task-allow)." >&2
    exit 1
  fi

  rm -f "${entitlements_tmp}"
}

installer_host_architectures () {
  if [[ "${UNIVERSAL}" == "1" ]]; then
    echo "arm64,x86_64"
    return
  fi
  case "$(uname -m)" in
    arm64) echo "arm64" ;;
    x86_64) echo "x86_64" ;;
    *) echo "arm64,x86_64" ;;
  esac
}

echo "==> Cleaning previous distribution artifacts"
rm -rf "${DIST_DIR}"
mkdir -p "${COMPONENT_PKG_DIR}" "${PAYLOAD_APP_ROOT}/Applications" "${LOCAL_DIST_DIR}"

if [[ -z "${APP_SIGN_IDENTITY_HASH}" ]]; then
  APP_SIGN_IDENTITY_HASH="$(resolve_codesign_identity_hash "${APP_SIGN_IDENTITY}" "application signing" "codesigning")"
fi

if [[ -z "${INSTALLER_SIGN_IDENTITY_HASH}" ]]; then
  INSTALLER_SIGN_IDENTITY_HASH="$(resolve_codesign_identity_hash "${INSTALLER_SIGN_IDENTITY}" "installer signing" "basic")"
fi

echo "==> Using application signing identity hash: ${APP_SIGN_IDENTITY_HASH}"
echo "==> Using installer signing identity hash: ${INSTALLER_SIGN_IDENTITY_HASH}"
echo "==> ${APP_NAME} ${VERSION} → ${PKG_FILE_BASENAME}-${VERSION}.pkg"

if [[ "${DIST_DIR}" == *"Mobile Documents"* || "${DIST_DIR}" == *"iCloud"* ]]; then
  echo "WARN: dist appears to be under iCloud Drive. Stapler/notary can fail (error 65)." >&2
fi

if [[ -z "${STAPLE_TMP_COPY}" ]]; then
  STAPLE_TMP_COPY=1
fi

cd "${PROJECT_DIR}"
GUI_BIN="$(build_gui_binary)"

echo "==> Assembling ${APP_NAME}.app"
assemble_app_bundle "${GUI_BIN}"
write_app_entitlements

cat > "${APP_COMPONENT_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
  <dict>
    <key>BundleHasStrictIdentifier</key>
    <true/>
    <key>BundleIsRelocatable</key>
    <false/>
    <key>BundleIsVersionChecked</key>
    <true/>
    <key>BundleOverwriteAction</key>
    <string>upgrade</string>
    <key>RootRelativeBundlePath</key>
    <string>Applications/${APP_NAME}.app</string>
  </dict>
</array>
</plist>
EOF

echo "==> Signing app (Developer ID, hardened runtime, timestamp)"
codesign --force --sign "${APP_SIGN_IDENTITY_HASH}" --timestamp --options runtime \
  --entitlements "${APP_ENTITLEMENTS}" "${APP_BUNDLE}"
codesign --verify --deep --strict --verbose=2 "${APP_BUNDLE}"

echo "==> Preflight: checking release entitlements"
assert_no_get_task_allow "${APP_BUNDLE}/Contents/MacOS/${APP_EXECUTABLE}"

echo "==> Staging installer payload"
ditto "${APP_BUNDLE}" "${PAYLOAD_APP_ROOT}/Applications/${APP_NAME}.app"
xattr -cr "${PAYLOAD_APP_ROOT}" 2>/dev/null || true

echo "==> Creating signed component package"
pkgbuild \
  --root "${PAYLOAD_APP_ROOT}" \
  --component-plist "${APP_COMPONENT_PLIST}" \
  --identifier "${PKG_ID_PREFIX}.app" \
  --version "${VERSION}" \
  --install-location "/" \
  --sign "${INSTALLER_SIGN_IDENTITY_HASH}" \
  "${APP_COMPONENT_PKG}"

HOST_ARCHS="$(installer_host_architectures)"
echo "==> Creating installer distribution (${HOST_ARCHS})"
cat > "${DIST_XML}" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>${INSTALLER_TITLE}</title>
  <options customize="always" require-scripts="true" hostArchitectures="${HOST_ARCHS}"/>
  <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
  <choices-outline>
    <line choice="choice.app"/>
    <line choice="choice.launchAfterInstall"/>
  </choices-outline>

  <choice
    id="choice.app"
    title="${APP_NAME}"
    description="Install /Applications/${APP_NAME}.app"
    start_selected="true"
    start_enabled="false">
    <pkg-ref id="${PKG_ID_PREFIX}.app"/>
  </choice>

  <choice
    id="choice.launchAfterInstall"
    title="Launch ${APP_NAME}"
    description="Open ${APP_NAME} when you close the installer."
    start_selected="true">
  </choice>

  <pkg-ref id="${PKG_ID_PREFIX}.app" version="${VERSION}" auth="Root">$(basename "${APP_COMPONENT_PKG}")</pkg-ref>

  <script><![CDATA[
function choiceIsSelected(choiceId) {
  try {
    return my.result.getChoice(choiceId).isSelected();
  } catch (e) {
    return false;
  }
}

function installationSucceeded(installResult) {
  if (typeof installResult !== 'undefined' && installResult !== 0 && installResult !== '0')
    return false;
  try {
    if (my.result.result && my.result.result !== 'success')
      return false;
  } catch (e) {}
  return true;
}

function installationDone(installResult) {
  if (!installationSucceeded(installResult))
    return;
  if (!choiceIsSelected('choice.launchAfterInstall'))
    return;
  system.run('/usr/bin/open', '/Applications/${APP_NAME}.app');
}
  ]]></script>

  <license file="License.txt" mime-type="text/plain" uti="public.plain-text"/>
</installer-gui-script>
EOF

INSTALLER_RESOURCES_DIR="${DIST_DIR}/installer_resources"
rm -rf "${INSTALLER_RESOURCES_DIR}"
mkdir -p "${INSTALLER_RESOURCES_DIR}"
if [[ -f "${LICENSE_SRC}" ]]; then
  echo "==> Adding LICENSE to installer"
  ditto "${LICENSE_SRC}" "${INSTALLER_RESOURCES_DIR}/License.txt"
else
  echo "WARN: ${LICENSE_SRC} not found; installer will have no license page."
  : > "${INSTALLER_RESOURCES_DIR}/License.txt"
fi

echo "==> Building signed product installer"
productbuild \
  --distribution "${DIST_XML}" \
  --package-path "${COMPONENT_PKG_DIR}" \
  --resources "${INSTALLER_RESOURCES_DIR}" \
  --sign "${INSTALLER_SIGN_IDENTITY_HASH}" \
  "${FINAL_PKG_PATH}"

chmod a+r "${FINAL_PKG_PATH}" 2>/dev/null || true
xattr -cr "${FINAL_PKG_PATH}" 2>/dev/null || true

copy_pkg_to_repo () {
  if [[ "${FINAL_PKG_PATH}" != "${LOCAL_FINAL_PKG_PATH}" ]]; then
    mkdir -p "${LOCAL_DIST_DIR}"
    ditto "${FINAL_PKG_PATH}" "${LOCAL_FINAL_PKG_PATH}"
    echo "  Copied to repo: ${LOCAL_FINAL_PKG_PATH}"
  fi
}

if [[ "${SKIP_NOTARIZATION}" == "1" ]]; then
  write_staple_status_file "signed installer built; notarization skipped (SKIP_NOTARIZATION=1)"
  copy_pkg_to_repo
  echo
  echo "==> SKIP_NOTARIZATION=1: skipping notarytool and stapler"
  echo "Done (signed Developer ID installer only)."
  echo "  Package: ${FINAL_PKG_PATH}"
  echo "For public distribution, run without SKIP_NOTARIZATION (needs notarytool keychain profile)."
  exit 0
fi

NOTARY_JSON="${DIST_DIR}/last_notary_submit.json"
echo "==> Submitting installer for notarization (same file that will be stapled — do not copy or modify the .pkg until after staple)"
if ! xcrun notarytool submit "${FINAL_PKG_PATH}" \
  --keychain-profile "${NOTARY_PROFILE}" \
  --team-id "${TEAM_ID}" \
  --wait \
  --output-format json > "${NOTARY_JSON}"; then
  echo "ERROR: notarytool submit failed." >&2
  cat "${NOTARY_JSON}" >&2 || true
  exit 1
fi

SUBMISSION_ID="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id") or "")' < "${NOTARY_JSON}")"
NOTARY_STATUS="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status") or "")' < "${NOTARY_JSON}")"
echo "    submission id: ${SUBMISSION_ID}"
echo "    status: ${NOTARY_STATUS}"

if [[ "${NOTARY_STATUS}" != "Accepted" ]]; then
  echo "ERROR: Notarization did not return Accepted. Fetching log:" >&2
  if [[ -n "${SUBMISSION_ID}" ]]; then
    xcrun notarytool log "${SUBMISSION_ID}" --keychain-profile "${NOTARY_PROFILE}" >&2 || true
  fi
  exit 1
fi

PKG_SHA256_AT_SUBMIT="$(pkg_sha256 "${FINAL_PKG_PATH}")"
echo "    pkg SHA256 at notarization (must not change before staple): ${PKG_SHA256_AT_SUBMIT}"
echo "    STAPLE_TMP_COPY=${STAPLE_TMP_COPY}  (default 1: staple via temp under ${STAPLE_TMP_DIR}; STAPLE_TMP_COPY=0 staples in place)"

NEXT_TIME="$(format_clock_time_from_now_secs "${POST_NOTARIZE_SLEEP_SEC}")"
echo "==> Waiting ${POST_NOTARIZE_SLEEP_SEC}s before first staple (Apple CDN ticket propagation; reduces stapler error 65)"
echo "    Next attempt <time: ${NEXT_TIME}>  (press C to abort stapling — exits gracefully; installer is already built and notarized)"
if ! sleep_with_optional_abort "${POST_NOTARIZE_SLEEP_SEC}"; then
  echo
  echo "=== Stapling aborted (C) before first attempt ===" >&2
  echo "Build result: installer signed and notarization Accepted; stapling not run." >&2
  echo "  Package: ${FINAL_PKG_PATH}" >&2
  echo "  Notary submission id: ${SUBMISSION_ID}" >&2
  echo "Run later: xcrun stapler staple \"${FINAL_PKG_PATH}\" && xcrun stapler validate \"${FINAL_PKG_PATH}\"" >&2
  write_staple_status_file "notarization Accepted; stapling aborted by user (C) before first staple attempt"
  copy_pkg_to_repo
  exit 0
fi

echo "==> Stapling notarization ticket onto installer"
echo "    ${FINAL_PKG_PATH}"
if [[ -t 0 ]] && [[ "${STAPLE_ABORT_ON_C}" == "1" ]]; then
  echo "    Press C to abort stapling (waits between attempts) — exits gracefully."
fi

staple_ok=0
for attempt in $(seq 1 "${STAPLE_RETRIES}"); do
  staple_flags=()
  if [[ "${STAPLE_VERBOSE}" == "1" ]] || { [[ "${STAPLE_VERBOSE_AFTER_FAIL}" == "1" ]] && [[ "${attempt}" -gt 1 ]]; }; then
    staple_flags+=(-v)
  fi

  now_hash="$(pkg_sha256 "${FINAL_PKG_PATH}")"
  if [[ "${now_hash}" != "${PKG_SHA256_AT_SUBMIT}" ]]; then
    echo "ERROR: Installer .pkg changed after notarization (SHA256 mismatch)." >&2
    echo "  At submit: ${PKG_SHA256_AT_SUBMIT}" >&2
    echo "  Now:       ${now_hash}" >&2
    echo "Do not edit, copy, or let cloud sync touch ${FINAL_PKG_PATH} until after stapling." >&2
    exit 1
  fi

  echo "==> Staple attempt ${attempt}/${STAPLE_RETRIES}"
  if staple_pkg_with_optional_tmp_copy "${FINAL_PKG_PATH}" ${staple_flags[@]+"${staple_flags[@]}"}; then
    staple_ok=1
    break
  else
    echo "WARN: stapler staple or validate failed (exit 65). Attempt ${attempt}/${STAPLE_RETRIES}."
    if [[ "${attempt}" -eq 1 ]]; then
      echo "    If you still see \"Could not validate ticket\" with STAPLE_TMP_COPY=1, try: STAPLE_TMP_DIR=/tmp $0 ..." >&2
      if [[ -n "${SUBMISSION_ID:-}" ]]; then
        echo "    Notary log: xcrun notarytool log \"${SUBMISSION_ID}\" --keychain-profile \"${NOTARY_PROFILE}\"" >&2
      fi
    fi
  fi

  if [[ "${attempt}" -lt "${STAPLE_RETRIES}" ]]; then
    WAIT_SECS="$(staple_wait_secs_after_failed_attempt "${attempt}")"
    NEXT_ATT_TIME="$(format_clock_time_from_now_secs "${WAIT_SECS}")"
    echo "Next attempt <time: ${NEXT_ATT_TIME}>  (in ${WAIT_SECS}s; press C to abort stapling — exits gracefully)"
    if ! sleep_with_optional_abort "${WAIT_SECS}"; then
      echo
      echo "=== Stapling aborted (C) ===" >&2
      echo "Build result: installer signed and notarization Accepted; stapling incomplete." >&2
      echo "  Package: ${FINAL_PKG_PATH}" >&2
      echo "  Notary submission id: ${SUBMISSION_ID}" >&2
      echo "Run later: xcrun stapler staple \"${FINAL_PKG_PATH}\" && xcrun stapler validate \"${FINAL_PKG_PATH}\"" >&2
      write_staple_status_file "notarization Accepted; stapling aborted by user (C) after attempt ${attempt}/${STAPLE_RETRIES}"
      copy_pkg_to_repo
      exit 0
    fi
  fi
done

if [[ "${staple_ok}" -ne 1 ]]; then
  echo "ERROR: Notarization accepted, but stapling still failed after ${STAPLE_RETRIES} attempts." >&2
  echo "Exit 65: either the ticket was not on Apple's CDN yet, or stapler downloaded a ticket that does not" >&2
  echo "match your local .pkg (common with iCloud/sync). Wait a few minutes and run:" >&2
  echo "  xcrun stapler staple \"${FINAL_PKG_PATH}\" && xcrun stapler validate \"${FINAL_PKG_PATH}\"" >&2
  echo "If it still fails with \"Could not validate ticket\", try STAPLE_TMP_DIR=/tmp." >&2
  echo "Expected SHA256 at submit: ${PKG_SHA256_AT_SUBMIT}" >&2
  if [[ -n "${SUBMISSION_ID}" ]]; then
    echo "Submission id (for Apple): ${SUBMISSION_ID}" >&2
  fi
  write_staple_status_file "notarization Accepted; stapling failed after ${STAPLE_RETRIES} attempts"
  exit 1
fi

write_staple_status_file "notarization Accepted; stapled and validated OK"
chmod a+r "${FINAL_PKG_PATH}" 2>/dev/null || true
copy_pkg_to_repo

echo
echo "Done."
echo "Installer (notarized + stapled; safe to copy or upload): ${FINAL_PKG_PATH}"
if [[ "${FINAL_PKG_PATH}" != "${LOCAL_FINAL_PKG_PATH}" ]]; then
  echo "Repo copy: ${LOCAL_FINAL_PKG_PATH}"
fi
