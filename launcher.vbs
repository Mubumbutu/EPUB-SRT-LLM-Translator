Dim objShell, objFSO, venvPython, appFile, baseDir

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

baseDir = objFSO.GetParentFolderName(WScript.ScriptFullName)

venvPython = baseDir & "\venv\Scripts\python.exe"
appFile = baseDir & "\app.py"

If Not objFSO.FileExists(venvPython) Then
    MsgBox "Virtual environment not found!" & vbCrLf & "Run install.bat first.", vbCritical, "Error"
    WScript.Quit 1
End If

If Not objFSO.FileExists(appFile) Then
    MsgBox "app.py not found!", vbCritical, "Error"
    WScript.Quit 1
End If

objShell.CurrentDirectory = baseDir
objShell.Run """" & venvPython & """ """ & appFile & """", 0, False