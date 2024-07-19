import Head from 'next/head'
import React, {useState, useEffect, useRef} from 'react'
import CreatableSelect from 'react-select/creatable';
import { InteractiveSegment, Point, Mask, Data, }
  from '../components/interactive_segment'
import * as utils from '@/utils';
import { PNG } from 'pngjs/browser';
const uiBasiclClassName = 'transition-all my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 ';
const uiActiveClassName = 'bg-blue-500 text-white';
const uiInactiveClassName = 'bg-white text-gray-400';

export const config = {
  basePath: '/sam',
  api: {
    bodyParser: false
  }
}

function Popup(text: string, timeout: number = 1000) {
  const popup = document.createElement('div')
  popup.className = 'fixed top-1/2 left-1/2 transform -translate-x-1/2 z-50 bg-white text-gray-500 rounded-xl px-4 py-2'
  popup.innerHTML = text
  document.body.appendChild(popup)
  setTimeout(() => {
    popup.remove()
  }, timeout)
}

function PageLoad() {
  return null;
}

function url_to_file(dataurl : any, filename : any) {
  var arr = dataurl.split(','),
      mime = arr[0].match(/:(.*?);/)[1],
      bstr = atob(arr[arr.length - 1]),
      n = bstr.length,
      u8arr = new Uint8Array(n);
  while(n--){
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, {type:mime});
}

function Workspace() {
  const [data, setData] = useState<Data | null>(null) // We can set this with the API to change the displayed image?
  const [mode, setMode] = useState<'click' | 'box' | 'everything'>('click')
  const [points, setPoints] = useState<Point[]>([])
  const [masks, setMasks] = useState<Mask[]>([])
  const [json_prompt, setJSONPrompt] = useState<string>('')
  const [text_prompt, setTextPrompt] = useState<string>('')
  const [processing, setProcessing] = useState<boolean>(false)
  const [ready, setBoxReady] = useState<boolean>(false)
  const controller = useRef<AbortController | null>()
  const [filename, setFilename] = useState('')
  const [imgx, setImageX] = useState('')
  const [imgy, setImageY] = useState('')
  const [img_loaded, setImageLoaded] = useState<boolean>(false)


  // Checks the display API for a file
  useEffect(() => {
    if (img_loaded) return

    // Get the parameter passed in the URL so that we can request the file for display
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search)
      let uuid = params.get('uuid')
      if (uuid !== null) {
        uuid = uuid.replace(/ /g, '+')
        setFilename(uuid)
        console.log('uuid:' + uuid)

        const fromData = new FormData()
        fromData.append('UUID', JSON.stringify({UUID: uuid}))

        // Get the image from the database
        fetch('/sam/api/get_file', {
          method: 'POST',
          body: fromData,
          signal: controller.current?.signal
        }).then((res) => {
          return res.json()
        }).then((res) => {
          if (res.code == 0) {
            setImageLoaded(true)
            const file_data = res.file
            const type = res.type
            const img = new Image()
            const file = url_to_file(`data:${type};base64,` + file_data, uuid)
            console.log(file)
            img.src = URL.createObjectURL(file)
            console.log(img.width)
            console.log(img.height)
            img.onload = () => {
              setImageX(img.width.toString())
              setImageY(img.height.toString())
              setData({
                width: img.width,
                height: img.height,
                file,
                img,
              })
            }
          }
          else {
            alert(`File with uuid ${uuid} not found in database.`)
          }
        })
      }
    }
  })

  // Handles all the point functions
  useEffect(() => {
    if (!data) return
    if (mode === 'click' && points.length > 0) {
      const fromData = new FormData()
      fromData.append('file', new File([data.file], 'image.png'))
      const points_list = points.map((p) => {
        return {
          x: Math.round(p.x),
          y: Math.round(p.y)
        }
      })
      //alert(JSON.stringify(points_list))
      //let points_list = [{"x":1132,"y":1597}]
      const points_labels = points.map((p) => p.label)
      fromData.append('points', JSON.stringify(
        { points: points_list, points_labels }
      ))
      controller.current?.abort()
      controller.current = new AbortController()
      setProcessing(true)
      fetch('/sam/api/point', {
        method: 'POST',
        body: fromData,
        signal: controller.current?.signal,
      }).then((res) => {
        return res.json()
      }).then((res) => {
        if (res.code == 0) {
          const maskData = res.data.map((mask: any) => {
            return mask
          })
          setMasks(maskData)
        }
      }).finally(() => {
        setProcessing(false)
      })
    }
    if (mode === 'box') {
      if (!ready) return
      if (points.length !== 2) return
      const fromData = new FormData()
      fromData.append('file', new File([data.file], 'image.png'))
      fromData.append('box', JSON.stringify(
        {
          x1: Math.round(points[0].x),
          y1: Math.round(points[0].y),
          x2: Math.round(points[1].x),
          y2: Math.round(points[1].y),
        }
      ))
      controller.current?.abort()
      controller.current = new AbortController()
      setProcessing(true)
      fetch('/sam/api/box', {
        method: 'POST',
        body: fromData,
        signal: controller.current?.signal
      }).then((res) => {
        return res.json()
      }).then((res) => {
        if (res.code == 0) {
          setPoints([])
          const maskData = res.data.map((mask: any) => {
            return mask
          })
          setMasks(maskData)
        }
      }).finally(() => {
        setProcessing(false)
        setBoxReady(false)
      })
    }
  }, [data, mode, points, ready])

  useEffect(() => {
    setPoints([])
    setMasks([])
    setProcessing(false)
    switch (mode) {
      case 'click':
        break
      case 'box':
        break
      case 'everything':
        break
    }
  }, [mode])

  //Temporarily disabled the button as it wasn't of use at the time
  const handleTextPrompt = () => {
    if (text_prompt === '' || !data) return
    const fromData = new FormData()

    fromData.append('file', new File([data.file], 'image.png'))
    fromData.append('prompt',
      JSON.stringify({ text: text_prompt }))
    controller.current?.abort()
    controller.current = new AbortController()
    setProcessing(true)
    fetch('/sam/api/clip', {
      method: 'POST',
      body: fromData,
      signal: controller.current?.signal
    }).then((res) => {
      setProcessing(false)
      return res.json()
    }).then((res) => {
      if (res.code == 0) {
        const maskData = res.data.map((mask: any) => {
          return mask
        })
        setMasks(maskData)
      }
    })
  }

  const handleEverything = () => {
    setMode('everything')
    if (!data) return
    const fromData = new FormData()
    fromData.append('file', new File([data.file], 'image.png'))
    controller.current?.abort()
    controller.current = new AbortController()
    setProcessing(true)
    fetch('/sam/api/everything', {
      method: 'POST',
      body: fromData,
      signal: controller.current?.signal
    }).then((res) => {
      setProcessing(false)
      return res.json()
    }).then((res) => {
      if (res.code == 0) {
        const maskData = res.data.map((mask: any) => {
          return mask
        })
        setMasks(maskData)
      }
    })
  }

  const handleDownload = () => {
    if (!data) return
    const fromData = new FormData()
    fromData.append('file', new File([data.file], 'image.png'))
    fromData.append('filename', JSON.stringify({
      filename: filename
    }))
    fromData.append('overlay_filename', JSON.stringify({
      filename: filename.split('.')[0]+"+overlay.jpg"
    }))
    fromData.append('imgx', JSON.stringify({
          x_dim: imgx
    }))
    fromData.append('imgy', JSON.stringify({
          y_dim: imgy
    }))
    fromData.append('points_filename', JSON.stringify({
      filename: filename.split('.')[0]+"+points.json"
    }))
    const points_list = points.map((p) => {
        return {
          x: Math.round(p.x),
          y: Math.round(p.y)
        }
      })
    const points_labels = points.map((p) => p.label)
      fromData.append('points', JSON.stringify(
        { points: points_list, points_labels }
      ))

    controller.current?.abort()
    controller.current = new AbortController()
    setProcessing(true)
    fetch('/sam/api/download', { //send it to the download api for the template site
      method: 'POST',
      body: fromData,
      signal: controller.current?.signal
    }).then((res) => {
      setProcessing(false)
      return res.json()
    }).then((res) => {
      if (res.code == 0) {
        alert('Image Overlay downloaded to server.')
      }
    })
  }

  //Should function the same as handleCopyPaste but with uploading a JSON file
  const handleUploadJSON = () => {
    if (!data) return
    const fromData = new FormData()

    const points_list = points.map((p) => {
      return {
        x: Math.round(p.x),
        y: Math.round(p.y)
      }
    })
    //alert(JSON.stringify(points_list))
    //let points_list = [{"x":1132,"y":1597}]
    const points_labels = points.map((p) => p.label)

    fromData.append('file', new File([data.file], 'image.png'))
    fromData.append('points', JSON.stringify({ points: points_list, points_labels }))
    controller.current?.abort()
    controller.current = new AbortController()
    setProcessing(true)
    fetch('/sam/api/copy-paste', {
      method: 'POST',
      body: fromData,
      signal: controller.current?.signal
    }).then((res) => {
      setProcessing(false)
      return res.json()
    }).then((res) => {
      if (res.code == 0) {
        const maskData = res.data.map((mask: any) => {
          return mask
        })
        setMasks(maskData)
      }
    })
  }

  const handleClick = () => {
    if (!data) return
    const fromData = new FormData()
    fromData.append('file', new File([data.file], 'image.png'))
    const points_list = points.map((p) => {
      return {
        x: Math.round(p.x),
        y: Math.round(p.y)
      }
    })
    const points_labels = points.map((p) => p.label)
    fromData.append('points', JSON.stringify(
        { points: points_list, points_labels }
    ))
    controller.current?.abort()
    controller.current = new AbortController()
    setProcessing(true)
    fetch('/sam/api/point', {
      method: 'POST',
      body: fromData,
      signal: controller.current?.signal,
    }).then((res) => {
      return res.json()
    }).then((res) => {
      if (res.code == 0) {
        const maskData = res.data.map((mask: any) => {
          return mask
        })
        setMasks(maskData)
      }
    }).finally(() => {
      setProcessing(false)
    })
  }

  const handleDicomImage = () => {
    setImageLoaded(true);
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*, .dcm, .dicom';

    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];

      if (file) {
        //console.log(file);
        setFilename(file.name.replace(/ /g, '+'));

        // Check if the file is a DICOM file based on its extension
        // @ts-ignore
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const isDicom = fileExtension === 'dcm' || fileExtension === 'dicom';

        if (isDicom) {
          // Push the file to the API endpoint
          const formData = new FormData();
          formData.append('file', file);

          const response = await fetch('/sam/api/dicom-to-png', {
            method: 'POST',
            body: formData,
          });

          const blob = await response.blob();
          console.log(blob);
          const img = new Image();
          const url = URL.createObjectURL(blob);
           img.src = url;
          img.onload = () => {
            URL.revokeObjectURL(url)
            setImageX(img.width.toString());
            setImageY(img.height.toString());
            setData({
              width: img.width,
              height: img.height,
              file,
              img,
            })
          }
        } else {
          // Handle image preview for non-DICOM files
          const img = new Image();
          img.src = URL.createObjectURL(file);
          img.onload = () => {
            setImageX(img.width.toString());
            setImageY(img.height.toString());
            setData({
              width: img.width,
              height: img.height,
              file,
              img,
            });
          };
        }
      }
    };

    input.click();
  };


  return (
    <div className="flex items-stretch justify-center flex-1 stage min-h-fit">
      <section className="flex-col hidden min-w-[225px] w-1/5 py-5 overflow-y-auto md:flex lg:w-72">
        <div className='shadow-[0px_0px_5px_5px_#00000024] rounded-xl mx-2'>
          <div className='p-4 pt-5'>
            <p className='text-lg font-semibold'>Tools</p>
              <div className={uiBasiclClassName}>
                <p>Interactive Mode</p>
                <div>
                  <button
                    className={
                      uiBasiclClassName +
                      (mode === 'click' ? uiActiveClassName : uiInactiveClassName)
                    }
                    onClick={() => { setMode('click') }} >
                    Click
                  </button>
                </div>
                <div>
                  <button
                    className={
                      uiBasiclClassName +
                      (mode === 'box' ? uiActiveClassName : uiInactiveClassName)
                    }
                    onClick={() => { setMode('box') }} >
                    Box
                  </button>
                </div>
                <div>
                  <button
                    className={
                      uiBasiclClassName +
                      (mode === 'everything' ? uiActiveClassName : uiInactiveClassName)
                    }
                    onClick={handleEverything} >
                    Everything
                  </button>
                </div>
              </div>
              <div className={uiBasiclClassName}>
                <p>Get Segment from Query</p>
                  <textarea className='w-full h-20 outline outline-gray-200 p-1'
                            onChange={(e) => {
                              setTextPrompt(e.target.value)
                            }}
                            value={text_prompt} />
                  <button
                      className='my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 bg-white hover:bg-blue-500 hover:text-white'
                      onClick={handleTextPrompt}
                  >
                    Send Query
                  </button>
              </div>
              <div className={uiBasiclClassName}>
                <p>Get Segment from JSON</p>
                <button
                    className='my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 bg-white hover:bg-blue-500 hover:text-white'
                    onClick={() => {
                      const input = document.createElement('input')
                      input.type = 'file'
                      input.accept = 'application/json'
                      input.onchange = (e) => {
                        const file = (e.target as HTMLInputElement).files?.[0]
                        if (file) {
                          const reader = new FileReader()
                          reader.onload = (event) => {
                            const contents = event.target?.result as string;
                            try {
                              const jsonData = JSON.parse(contents);
                              setPoints(jsonData.points)
                              handleUploadJSON()
                              console.log(jsonData)
                              // Perform further actions with the parsed JSON data here
                            } catch (error) {
                              console.error('Error parsing JSON file:', error)
                            }
                          }
                          reader.readAsText(file)
                        }
                      }
                      input.click()
                    }}
                >
                  Upload Points JSON File
                </button>
              </div>
              {masks.length > 0 && (
                <div className={uiBasiclClassName}>
                  <p>Save Progress</p>
                  <button
                    className={uiBasiclClassName}
                    onClick={handleDownload}
                  >
                    Download Results
                  </button>
                </div>
              )}
            <div className={uiBasiclClassName}>
              <p>Interactive Setting</p>
              <button
                className='false my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
                onClick={() => {
                  setPoints([])
                  setMasks([])
                  setBoxReady(false)
                  setProcessing(false)
                }} >
                Clean Segment
              </button>
              <button
                className='false my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
                onClick={() => {
                  setData(null)
                  setPoints([])
                  setMasks([])
                  setMode('click')
                }}>
                Clean All
              </button>
              <button
                  className='false my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
                  onClick={() => {
                    points.pop()
                    setMasks([])
                    handleClick()
                  }}>
                Undo Last Point
              </button>
            </div>
          </div>
        </div>
      </section >
      {
        (data && img_loaded ?
          (<div className="flex flex-1 flex-col max-w-[1080px] m-auto my-2 md:px-12 md:py-9" >
            <InteractiveSegment
              data={data} mode={mode} processing={processing}
              points={points} setPoints={setPoints} masks={masks}
              ready={ready} setBoxReady={setBoxReady} basePath={'/sam'}/>
            {processing && (
              <div className=" left-0 w-full flex items-center bg-black bg-opacity-50">
                <div className="flex flex-col items-center justify-center w-full h-full">
                  <div className="text-white text-2xl">Processing</div>
                  <div className='flex flex-row justify-center'>
                    <div className='w-2 h-2 bg-white rounded-full animate-bounce mx-1'></div>
                    <div className='w-2 h-2 bg-white rounded-full animate-bounce mx-1'></div>
                    <div className='w-2 h-2 bg-white rounded-full animate-bounce mx-1'></div>
                  </div>
                  <div className="text-white text-sm">Please wait a moment</div>
                </div>
              </div>
            )
            }
          </div>) :
            (
                <div className="flex flex-1 flex-col max-w-[1080px] m-auto my-2 md:px-12 md:py-9">
            <div
              className={
                "flex flex-col items-center justify-center w-full h-96 border-2 border-dashed border-gray-400 rounded-lg " +
                "hover:border-blue-500 hover:bg-blue-50 hover:text-blue-500" +
                "focus-within:border-blue-500 focus-within:bg-blue-50 focus-within:text-blue-500"
              }
              onDragOver={(e) => {
                e.preventDefault()
              }}
              onDrop={(e) => {
                e.preventDefault()
                const file = e.dataTransfer.files[0]
                if (file) {
                  setImageLoaded(true)
                  setFilename(file.name.replace(/ /g, '+'))
                  const img = new Image()
                  img.src = URL.createObjectURL(file)
                  img.onload = () => {
                    setImageX(img.width.toString())
                    setImageY(img.height.toString())
                    setData({
                      width: img.width,
                      height: img.height,
                      file,
                      img,
                    })
                  }
                }
              }}
            >
              <p className="text-sm text-gray-400 md:visible sm:invisible">Drag and drop your image here</p>
              <p className="text-sm text-gray-400">or</p>
              <button //this would be a good place to put an image converter that changes DICOM to png. This can be done using pngjs
                className="transition-all false max-h-[40px] my-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 false false"
                onClick={handleDicomImage}
              >
                Upload a DICOM or Image
              </button>
            </div>
          </div>))
      }
      <section className="hidden w-full absolute bottom-0 max-md:inline-block">
        <div className='transition-all m-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'>
          <p>Interactive Mode</p>
          <button
            className={
              'transition-all m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 ' +
              (mode === 'click' ? uiActiveClassName : uiInactiveClassName)
            }
            onClick={() => { setMode('click') }} >
            Click
          </button>
          <button
            className={
              'transition-all m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 ' +
              (mode === 'everything' ? uiActiveClassName : uiInactiveClassName)
            }
            onClick={handleEverything} >
            Everything
          </button>
        </div>

        <div className='transition-all m-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'>
          <textarea className='w-full h-20 outline outline-gray-200 p-1'
            onChange={(e) => {
              setTextPrompt(e.target.value)
            }}
            value={text_prompt} />
          <button
            className='m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200 bg-white hover:bg-blue-500 hover:text-white'
            onClick={handleTextPrompt}
          >CLIP Send</button>
        </div>
        {masks.length > 0 && (
          <div className='transition-all m-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'>
            <p>Segment</p>
            <button
              className='transition-all m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
              onClick={(e) => {
                var datastr = "data:text/json;charset=utf-8," + encodeURIComponent(
                  JSON.stringify({
                    masks: masks,
                    points: points,
                  }));
                var downloadAnchorNode = document.createElement('a');
                downloadAnchorNode.setAttribute("href", datastr);
                downloadAnchorNode.setAttribute("download", "masks.json");
                document.body.appendChild(downloadAnchorNode); // required for firefox
                downloadAnchorNode.click();
                downloadAnchorNode.remove();
              }}
            >Download Result</button>
            <button
              className='transition-all overflow m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
              onClick={(e) => {
                navigator.clipboard.writeText(JSON.stringify({
                  masks: masks,
                  points: points,
                }))
                Popup('Copied', 1000)
              }}
            >
              Copy Result
            </button>
          </div>
        )}
        <div className='transition-all overflow m-2 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'>
          <button
            className='false m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
            onClick={() => {
              setPoints([])
              setMasks([])
              setBoxReady(false)
              setProcessing(false)
            }} >
            Clean Segment
          </button>
          <button
            className='false m-1 rounded-xl px-4 py-2 cursor-pointer outline outline-gray-200'
            onClick={() => {
              setData(null)
              setPoints([])
              setMasks([])
              setMode('click')
            }} >
            Clean All
          </button>
        </div>
      </section>
    </div >
  )
}

export default function Home(p: any[]) {

  return (
    <>
      <Head>
        <title>Image Segment</title>
        <meta name="description" content="Image Segment" />
      </Head>
      <main className="flex-col min-h-full">
        <p className="m-2 text-xl font-bold leading-tight md:mx-6 lg:text-2xl">Segment Medical Images in Browser</p>
        <div className="flex items-center border-b-[1px] pt-3 pb-3"></div>
        <Workspace />
      </main>
    </>
  )
}
