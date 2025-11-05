Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir
'
' Run the Windows launcher hidden (no console window)
On Error Resume Next
WshShell.Run "cmd /c launch_windows.bat", 0, False
If Err.Number <> 0 Then
  Err.Clear
  ' Fallback: try pythonw directly if batch is missing
  WshShell.Run "pythonw.exe launch.py", 0, False
End If

