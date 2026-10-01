; Inno Setup 6 script for Medical Visit Assistant (per-user install, no admin rights).
; Build: iscc /DAppVersion=0.1.0 installer\mva.iss  (after PyInstaller produced dist\MedicalVisitAssistant)

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "Medical Visit Assistant"
#define AppExe "MedicalVisitAssistant.exe"

[Setup]
AppId={{8E0B6C47-3E0F-4C8E-9C61-2A3C5D1B7F21}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=MVA
DefaultDirName={localappdata}\Programs\MedicalVisitAssistant
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=MedicalVisitAssistant-{#AppVersion}-Setup
SetupIconFile=..\packaging\mva.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Ярлыки:"

[Files]
Source: "..\dist\MedicalVisitAssistant\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\THIRD_PARTY_LICENSES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\PRIVACY.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; Removes the optional «Запускать вместе с Windows» entry on uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueName: "MedicalVisitAssistant"; Flags: dontcreatekey uninsdeletevalue

[Run]
Filename: "{app}\{#AppExe}"; Description: "Запустить {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Temporary visit audio is always removed. Settings and the downloaded model are kept
; (unless the user agrees below) so reinstalling/updating does not re-download 450 MB.
Type: filesandordirs; Name: "{localappdata}\MedicalVisitAssistant\temp"
Type: filesandordirs; Name: "{localappdata}\MedicalVisitAssistant\logs"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\MedicalVisitAssistant');
    if DirExists(DataDir) and (not UninstallSilent) then
      if MsgBox('Удалить также настройки и скачанную модель GigaAM (около 450 МБ)?',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
