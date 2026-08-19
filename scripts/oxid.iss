; Inno Setup 6 script for Oxid.
; Defines are passed by scripts/release_windows_installer.ps1:
;   AppVersion, SourceExe, SourceLicense, SourceLogo, SourceIcon, OutputDir

#ifndef AppName
  #define AppName "Oxid"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.2"
#endif
#ifndef AppPublisher
  #define AppPublisher "Oxid"
#endif
#ifndef SourceExe
  #define SourceExe "..\target\x86_64-pc-windows-msvc\release\Oxid.exe"
#endif
#ifndef SourceLicense
  #define SourceLicense "..\LICENSE"
#endif
#ifndef SourceLogo
  #define SourceLogo "..\src\img\logo.png"
#endif
#ifndef SourceIcon
  #define SourceIcon "..\build\Oxid.ico"
#endif
#ifndef OutputDir
  #define OutputDir "..\build\dist"
#endif
#ifndef OutputBase
  #define OutputBase "Oxid-" + AppVersion + "-setup"
#endif

[Setup]
AppId={{9E4C2B1A-7F53-4D8E-A6C1-0B2E8F4A91D3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=Copyright (C) {#AppPublisher}. Licensed under GPL-3.0-or-later.
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile={#SourceLicense}
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBase}
SetupIconFile={#SourceIcon}
UninstallDisplayIcon={app}\{#AppName}.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
ChangesAssociations=yes
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "oxidprojassoc"; Description: "Associate Oxid Project files (.oxidProj)"; GroupDescription: "File associations:"; Flags: checkedonce

[Files]
Source: "{#SourceExe}"; DestDir: "{app}"; DestName: "Oxid.exe"; Flags: ignoreversion
Source: "{#SourceLicense}"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#SourceLogo}"; DestDir: "{app}"; DestName: "logo.png"; Flags: ignoreversion
Source: "{#SourceIcon}"; DestDir: "{app}"; DestName: "Oxid.ico"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\Oxid.exe"; IconFilename: "{app}\Oxid.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Oxid.exe"; IconFilename: "{app}\Oxid.ico"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.oxidProj"; ValueType: string; ValueName: ""; ValueData: "Oxid.Project"; Flags: uninsdeletevalue; Tasks: oxidprojassoc
Root: HKA; Subkey: "Software\Classes\.c41proj"; ValueType: string; ValueName: ""; ValueData: "Oxid.Project"; Flags: uninsdeletevalue; Tasks: oxidprojassoc
Root: HKA; Subkey: "Software\Classes\Oxid.Project"; ValueType: string; ValueName: ""; ValueData: "Oxid Project"; Flags: uninsdeletekey; Tasks: oxidprojassoc
Root: HKA; Subkey: "Software\Classes\Oxid.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Oxid.ico"; Tasks: oxidprojassoc
Root: HKA; Subkey: "Software\Classes\Oxid.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Oxid.exe"" ""%1"""; Tasks: oxidprojassoc

[Run]
Filename: "{app}\Oxid.exe"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
