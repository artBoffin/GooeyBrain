const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;
const Tray = electron.Tray;

// Keep a global reference of the window object
var mainWindow = null;

// Quit when all windows are closed.
app.on('window-all-closed', function() {
// if (process.platform != 'darwin') {
    app.quit();
// }
});


app.on('ready', function() {
  // call python?
  var subpy = require('child_process').spawn('python', [__dirname + '/main.py']);

  var rq = require('request-promise');
  var mainAddr = 'http://localhost:8000';

  const appIcon = new Tray(__dirname + '/app/static/imgs/ArtificialFictionBrain.png');

  var openWindow = function(){
    // Create the browser window.
    mainWindow = new BrowserWindow({width: 800, height: 600, title:'GooeyBrain', icon:__dirname + '/app/static/imgs/ArtificialFictionBrain.png' });
    mainWindow.loadURL('http://localhost:8000');

    // Open the devtools.
    mainWindow.webContents.openDevTools();

    mainWindow.on('closed', function() {
      mainWindow = null;
      // kill python
      subpy.kill('SIGINT');
    });
     app.on('exit', function () {
          subpy.kill();
      });
  };

  var sendStdOut = function() {
      subpy.stdout.on('data', function(data) {
          console.log('stdout: ' + data);
          mainWindow.webContents.send('info-log' , {msg:String(data)});
      });
      subpy.stderr.on('data', function(data) {
          console.log('stderr: ' + data);
          mainWindow.webContents.send('info-log' , {msg:String(data)});
      });
      subpy.on('close', function(code) {
          console.log('closing code: ' + code);
      });
  }

  var startUp = function(){
    rq(mainAddr)
      .then(function(htmlString){
        console.log('server started!');
        openWindow();
        sendStdOut();
      })
      .catch(function(err){
        //console.log('waiting for the server start...');
        startUp();
      });
  };

  // fire!
  startUp();
});

const ipc = require('electron').ipcMain;
const dialog = require('electron').dialog;

ipc.on('open-dir-dialog', function (event) {
  dialog.showOpenDialog({
      properties: ['openFile', 'openDirectory']
  }, function (files) {
    if (files) {
        console.log(files);
        event.sender.send('selected-directory', files);
    }
  })
});

ipc.on('open-file-dialog', function (event) {
    dialog.showOpenDialog({
        properties: ['openFile']
    }, function (files) {
        if (files) {
            console.log(files);
            event.sender.send('selected-file', files);
        }
    })
});


