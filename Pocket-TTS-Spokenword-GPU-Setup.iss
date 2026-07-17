; Pocket-TTS-Spokenword GPU - Lightweight Installer
; Download Inno Setup: https://jrsoftware.org/isdl.php
; Installer downloads dependencies during install (requires internet)

#define MyAppName "Pocket-TTS-Spokenword GPU"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "gstock99"
#define MyAppURL "https://github.com/gstock99/Pocket-TTS-Spokenword-GPU"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=installer_output
OutputBaseFilename=Pocket-TTS-Spokenword-GPU-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Main source code
Source: "pocket_tts\*"; DestDir: "{app}\pocket_tts"; Flags: ignoreversion recursesubdirs createallsubdirs

; Python executables and DLLs
Source: "python\*.exe"; DestDir: "{app}\python"; Flags: ignoreversion
Source: "python\*.pyd"; DestDir: "{app}\python"; Flags: ignoreversion
Source: "python\*.dll"; DestDir: "{app}\python"; Flags: ignoreversion

; Python Scripts (pip etc)
Source: "python\Scripts\*"; DestDir: "{app}\python\Scripts"; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements_windows.txt"; DestDir: "{app}"; Flags: ignoreversion

; Launchers
Source: "launch_gui.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "launcher.pyw"; DestDir: "{app}"; Flags: ignoreversion

; Utility scripts
Source: "checkpoint.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "restore.bat"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name: "{app}\Output"
Name: "{app}\ebooks_tts"
Name: "{app}\sample_voices"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "launch_gui.py"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "launch_gui.py"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
Filename: "{app}\python\python.exe"; Parameters: "-m pip install --upgrade pip"; StatusMsg: "Upgrading pip..."; Flags: runhidden
Filename: "{app}\python\python.exe"; Parameters: "-m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"; StatusMsg: "Installing PyTorch with CUDA (~2GB download, please wait)..."; Flags: runhidden
Filename: "{app}\python\python.exe"; Parameters: "-m pip install -r requirements_windows.txt"; StatusMsg: "Installing remaining dependencies..."; Flags: runhidden
Filename: "{app}\python\python.exe"; Parameters: "-c ""from huggingface_hub import snapshot_download; snapshot_download('kyutai/pocket-tts')"""; StatusMsg: "Downloading AI model (~1GB, please wait)..."; Flags: runhidden
Filename: "{app}\launch_gui.py"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\Output"
Type: filesandordirs; Name: "{app}\ebooks_tts"
Type: filesandordirs; Name: "{app}\sample_voices"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: files; Name: "{app}\*.log"
